from HandRecog import *
from three import *
import asyncio
import websockets
import json

SERVER_URI = "ws://10.203.94.66:8765"

async def send_angles(angles):
    try:
        async with websockets.connect(SERVER_URI) as websocket:
            # Add 90 degrees to first 3 angles and append 3 more angles of 30 degrees
            adjusted_angles = [int(np.degrees(angle)) + 90 for angle in angles[:3]] + [30, 30, 30]
            data = {"servo_angles": adjusted_angles}
            message = json.dumps(data)
            await websocket.send(message)
            print(f"Sent angles to servos: {message}")
    except Exception as e:
        print(f"Error sending angles: {e}")

def main():
        ctime = 0
        ptime = 0
        frame_count = 0  # Add counter to track frames
        #Setting values for important variables for later. 

        cap = cv2.VideoCapture(0)
        #Takes video input from the first deteted camera. 
        
        detector = HandTrackingDynamic()
            # This declares detector to be an object of the HandTrackingDyanmic class, which gives it access to all the functions (methods) above.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        # Initialize robotic arm parameters outside loop
        k = np.array([[0,0,1], [0,0,1], [0,0,1]])  # Joint rotation axes
        
        a1, a2, a3, a4, a5 = 4.7, 5.9, 5.4, 6.0, 5.0
        t = np.array([[0,0,0], [a2,0,a1], [0,a4,0]])  # Joint translations
        
        p_eff_2 = [7,a3+2.25,0.25]  # End effector position
        k_c = RoboticArm(k,t)
        q_current = np.array([0,0,0])  # Starting angles

        # Reference point
        reference_point = np.array([550, 275, 1200])

        while True:
            ret, frame = cap.read()
                #take camera input

            frame = detector.processAndCorrectView(frame)
                #process camera input and flip view
            frame = detector.drawHandLandmarks(frame)
                #draw intial la ndmark drawings
            lmsList = detector.findAndMark_Positions(frame)
                #Determine pixel positions and do secondary landmark drawing.

            ctime = time.time()
            fps = 1/(ctime-ptime)
            ptime = ctime
                #ctime is the time at which the loop was last ran. ptime stores the previous ctime. 
                #Hence, the FPS actually refers to how often landmark (knuckle) locations are calculated per second. 

            fontSize = 1.2
            fontThickness = 2

            cv2.putText(frame, ("FPS: " + str(int(fps))), (5,30), cv2.FONT_HERSHEY_PLAIN, fontSize, (0,255,0), fontThickness)
                #On screen FPS counter. Second argument is text to be displayed, third is location and the rest is font/color/formatting.

            if len(lmsList[0]) != 0:
                # This if statement is necessary because without it, the program kills itself when your hand isn't on screen lol. 
                frame = detector.markOrientation(frame)
                rotation, _ = detector.findRotation(frame)
                forwardTilt, sidewaysTilt = detector.findTilt(frame)
                centerOfMassWithFingers, centerOfMassNoFingers = detector.findAndMarkCenterOfMass(frame)
                fingers, handMsg, _ = detector.findFingersOpen()

                #print(detector.lmsList[0][3]*5e5)
                #print("Fingers Open: ", fingers, (sum(fingers[0:5])), "  ", verticalDistance, handHorizontalOrientation, handVerticalOrientation, horizontalDistance, maxDistance, rotation)
                    # Output finger states and other info to console
                
                cv2.putText(frame, ("Fingers Open: " + str(fingers) + " " + str(sum(fingers[0:5])) + "  Hand is " + handMsg), (5,60), cv2.FONT_HERSHEY_PLAIN, fontSize, (0,255,0), fontThickness)
                    #On-screen hand status. 
                cv2.putText(frame, ("Rotation: " + str(rotation)), (5,90), cv2.FONT_HERSHEY_PLAIN, fontSize, (0,255,0), fontThickness)
                cv2.putText(frame, ("Forward Tilt: " + str(forwardTilt) + "  Sideways Tilt:" + str(sidewaysTilt)), (5,120), cv2.FONT_HERSHEY_PLAIN, fontSize, (0,255,0), fontThickness)
                cv2.putText(frame, ("Center of Mass:"), (5,160), cv2.FONT_HERSHEY_PLAIN, fontSize, (0,255,0), fontThickness)
                cv2.putText(frame, ("  With Fingers: " + str(centerOfMassWithFingers[1:])), (5,190), cv2.FONT_HERSHEY_PLAIN, fontSize, (0,255,0), fontThickness)
                cv2.putText(frame, ("  Without Fingers: " + str(centerOfMassNoFingers[1:])), (5,220), cv2.FONT_HERSHEY_PLAIN, fontSize, (0,255,0), fontThickness)

                # Only calculate angles every 2 frames
                if frame_count % 2 == 0:
                    # Calculate displacement from reference point using only x,y,z coordinates
                    displacement = np.array(centerOfMassNoFingers[1:4]) - reference_point

                    # Calculate end effector position based on displacement
                    endeffector_goal_position = np.array([
                        displacement[2]/400,  # Scale down displacement 
                        displacement[0]/200,
                        displacement[1]/800
                    ])
                    final_q = k_c.pseudo_inverse(q_current, p_eff_N=p_eff_2, goal_position=endeffector_goal_position, max_steps=500)
                    
                    # Update current angles for next iteration
                    #q_current = final_q
                    
                    print('\n\nFinal Joint Angles in Degrees')
                    print(f'Joint 1: {np.degrees(final_q[0])} , Joint 2: {np.degrees(final_q[1])}, Joint 3: {np.degrees(final_q[2])}')
                    
                    # Check if any angles are outside -90 to 90 degrees before sending
                    angles_in_degrees = [np.degrees(angle) for angle in final_q]
                    if all(-90 <= angle <= 90 for angle in angles_in_degrees):
                        # Send angles to servos only if all angles are within range
                        asyncio.run(send_angles(final_q))
                        print()
                    else:
                        print("Warning: Some angles out of -90 to 90 degree range, not sending to servos")

            else: 
                cv2.putText(frame, ("Awaiting Hand..    ."), (5,70), cv2.FONT_HERSHEY_PLAIN, 2, (74,26,255), 2)
                    #On screen FPS counter. Second argument is text to be displayed, third is location and the rest is font/color/formatting.

            frame_count += 1  # Increment frame counter

            if cv2.waitKey(1) == ord('x'):
                break
                    #break condition: if x is pressed, stops loop
            
            cv2.imshow('Hand Movement Interpreter', frame)
                #Opens a window with the name Hand Movement Interpreter and displays the result of running the above code on the camera input.

if __name__ == "__main__":
            main()

        #These two lines just make sure that main() doesnt run unless this script is run directly. Prevents it from running unintentionally if this script is imported into another program.