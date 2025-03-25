import numpy as np
import cv2
import time
import asyncio
import websockets
import json
from HandRecog import HandTrackingDynamic

SERVER_URI = "ws://10.203.94.66:8765"

async def send_angles(angles, handIsClosed, servo4_angle, servo5_angle):
    try:
        async with websockets.connect(SERVER_URI) as websocket:
            # Convert to degrees and add 90 degree offset for servo 1
            servo_angles = [
                int(np.degrees(angles[0])) + 90,  # t1 offset
                int(np.degrees(angles[1])),       # t2
                180-int(np.degrees(angles[2])+150),      # t3 (negative)
                servo4_angle, servo5_angle,          # Fixed angle for servo 4, variable for servo 5
                0 if handIsClosed else 120      # Servo 6 based on hand state
            ]
            data = {"servo_angles": servo_angles}
            message = json.dumps(data)
            await websocket.send(message)
            print(f"Sent angles to servos: {message}")
    except Exception as e:
        print(f"Error sending angles: {e}")

def main():
    servo1_angle = 90
    servo2_angle = 90
    servo3_angle = 90
    servo4_angle = 90
    servo4_angle = 90
    servo5_angle = 70

    ctime = 0
    ptime = 0
    frame_count = 0

    cap = cv2.VideoCapture(0)
    detector = HandTrackingDynamic()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Initialize robotic arm parameters
    a1, a2, a3 = 2, 4.75, 7.5  # Link lengths
    Z0 = 3.75  # Base height

    # Reference point for hand tracking
    #reference_point = np.array([550, 250, 1900])
    reference_point = np.array([550, 240, 1400])  

    while True:
        ret, frame = cap.read()
        frame = detector.processAndCorrectView(frame)
        frame = detector.drawHandLandmarks(frame)
        lmsList = detector.findAndMark_Positions(frame)

        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime

        fontSize = 1.2
        fontThickness = 2
        cv2.putText(frame, ("FPS: " + str(int(fps))), (5,30), cv2.FONT_HERSHEY_PLAIN, fontSize, (0,255,0), fontThickness)

        if len(lmsList[0]) != 0:
            frame = detector.markOrientation(frame)
            rotation, _ = detector.findRotation(frame)
            forwardTilt, sidewaysTilt = detector.findTilt(frame)
            forwardTilt = max(min(forwardTilt, 0.5), -0.5)  # Clamp forwardTilt between -0.5 and 0.5
            rotation = max(min(rotation, 0.5), -0.5)  # Clamp rotation between -0.5 and 0.5
            
            # Update servo5_angle based on forwardTilt, keeping within bounds
            if forwardTilt >= 0.35 and servo5_angle > 20:
                servo5_angle = max(40, servo5_angle - 4)  # Decrement but don't go below 20
            elif forwardTilt <= -0.35 and servo5_angle < 160:
                servo5_angle = min(140, servo5_angle + 4)  # Increment but don't exceed 160

            # Update servo4_angle based on rotation, keeping within bounds
            if rotation >= 0.35 and servo4_angle > 20:
                servo4_angle = max(20, servo4_angle - 2)  # Decrement but don't go below 20
            elif rotation <= -0.35 and servo4_angle < 160:
                servo4_angle = min(160, servo4_angle + 2)  # Increment but don't exceed 160
            centerOfMassWithFingers, centerOfMassNoFingers = detector.findAndMarkCenterOfMass(frame)
            fingers, handMsg, handIsClosed = detector.findFingersOpen()

            # Calculate displacement from reference point
            displacement = np.array(centerOfMassNoFingers[1:4]) - reference_point

            # Map hand position to robot target coordinates
            #X = 7 - displacement[2]/125
            #Y = (-displacement[0]/135)**3
            #Z = 5 + (displacement[1]/100)**3
            X = 7 - displacement[2]/125
            Y = (-displacement[0]/80)
            Z = 6 + (displacement[1]/70)

            try:
                # Calculate inverse kinematics
                # Compute X0 and Y0 using the base radius
                A = np.sqrt(a1**2 - Z0)
                t1 = np.arctan2(Y, X)  # Base rotation
                
                # Check t1 constraint (-90° to 90°)
                if not (-np.pi/2 <= t1 <= np.pi/2):
                    raise ValueError("t1 angle outside valid range [-90°, 90°]")
                    
                X0 = A * np.cos(t1)  # Base joint X position
                Y0 = A * np.sin(t1)  # Base joint Y position

                # Compute intermediate value D
                D = ((X - X0) ** 2 + (Y - Y0) ** 2 + (Z - Z0) ** 2 - a2 - a3) / (2 * a2 * a3)

                # Ensure D is within valid range for acos
                if np.abs(D) > 1:
                    raise ValueError("Target position is unreachable")

                # Compute t3 (elbow-up and elbow-down)
                t3_up = np.arctan2(np.sqrt(1 - D**2), D)
                t3_down = np.arctan2(-np.sqrt(1 - D**2), D)

                # Compute t2 for both configurations
                phi1 = np.arctan2(Z - Z0, np.sqrt((X - X0) ** 2 + (Y - Y0) ** 2))
                
                # For t3_up
                phi2_up = np.arctan2(a3 * np.sin(t3_up), a2 + a3 * np.cos(t3_up))
                t2_up = phi1 - phi2_up
                
                # For t3_down  
                phi2_down = np.arctan2(a3 * np.sin(t3_down), a2 + a3 * np.cos(t3_down))
                t2_down = phi1 - phi2_down

                # Check angle constraints and select valid solution
                angles = None

                if (0 <= t2_up <= np.pi and -5*np.pi/6 <= t3_up <= np.pi/6):
                    angles = (t1, t2_up, t3_up)
                elif (0 <= t2_down <= np.pi and -5*np.pi/6 <= t3_down <= np.pi/6):
                    angles = (t1, t2_down, t3_down)
                
                if angles is None:
                    raise ValueError("No solutions found within angle constraints")

                # Convert angles to degrees
                new_angles = (np.degrees(angles[0]), np.degrees(angles[1]), np.degrees(angles[2]))

                # Check if angles have changed significantly (> 2 degrees)
                should_update = False
                if 'servo1_angle' not in locals() or abs(new_angles[0] - servo1_angle) > 2 or \
                   abs(new_angles[1] - servo2_angle) > 2 or abs(new_angles[2] - servo3_angle) > 2:
                    should_update = True
                    servo1_angle = new_angles[0]
                    servo2_angle = new_angles[1]
                    servo3_angle = new_angles[2]

                print('\nCalculated Joint Angles in Degrees:')
                print(f't1: {new_angles[0]:.2f}°')
                print(f't2: {new_angles[1]:.2f}°')
                print(f't3: {new_angles[2]:.2f}°')

                # Only send angles if they've changed significantly
                if should_update:
                    asyncio.run(send_angles(angles, handIsClosed, servo4_angle, servo5_angle))

            except ValueError as e:
                print(f"Inverse kinematics error: {e}")

            cv2.putText(frame, ("Fingers Open: " + str(fingers) + " " + str(sum(fingers[0:5])) + "  Hand is " + handMsg), (5,60), cv2.FONT_HERSHEY_PLAIN, fontSize, (0,255,0), fontThickness)
            cv2.putText(frame, ("Rotation: " + str(rotation)), (5,90), cv2.FONT_HERSHEY_PLAIN, fontSize, (0,255,0), fontThickness)
            cv2.putText(frame, ("Forward Tilt: " + str(forwardTilt) + "  Sideways Tilt:" + str(sidewaysTilt)), (5,120), cv2.FONT_HERSHEY_PLAIN, fontSize, (0,255,0), fontThickness)
            cv2.putText(frame, ("Center of Mass:"), (5,160), cv2.FONT_HERSHEY_PLAIN, fontSize, (0,255,0), fontThickness)
            cv2.putText(frame, ("  With Fingers: " + str(centerOfMassWithFingers[1:])), (5,190), cv2.FONT_HERSHEY_PLAIN, fontSize, (0,255,0), fontThickness)
            cv2.putText(frame, ("  Without Fingers: " + str(centerOfMassNoFingers[1:])), (5,220), cv2.FONT_HERSHEY_PLAIN, fontSize, (0,255,0), fontThickness)
        else:
            cv2.putText(frame, ("Awaiting Hand..."), (5,70), cv2.FONT_HERSHEY_PLAIN, 2, (74,26,255), 2)

        frame_count += 1

        if cv2.waitKey(1) == ord('x'):
            break

        cv2.imshow('Hand Movement Interpreter', frame)

if __name__ == "__main__":
    main()