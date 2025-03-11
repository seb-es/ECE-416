import numpy as np
import cv2
import time
import asyncio
import websockets
import json
from HandRecog import HandTrackingDynamic

SERVER_URI = "ws://10.203.94.66:8765"

async def send_angles(angles, handIsClosed):
    try:
        async with websockets.connect(SERVER_URI) as websocket:
            # Convert to degrees and add 90 degree offset for servo 1
            servo_angles = [
                int(np.degrees(angles[0])) + 90,  # t1 offset
                int(np.degrees(angles[1])),       # t2
                180-int(np.degrees(angles[2])+150),      # t3 (negative)
                0, 90,                          # Fixed angles for servos 4-5
                0 if handIsClosed else 120      # Servo 6 based on hand state
            ]
            data = {"servo_angles": servo_angles}
            message = json.dumps(data)
            await websocket.send(message)
            print(f"Sent angles to servos: {message}")
    except Exception as e:
        print(f"Error sending angles: {e}")

def main():
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
    reference_point = np.array([550, 250, 1900])

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
            _, centerOfMassNoFingers = detector.findAndMarkCenterOfMass(frame)
            handIsClosed = detector.findFingersOpen()[2]

            if frame_count % 2 == 0:
                # Calculate displacement from reference point
                displacement = np.array(centerOfMassNoFingers[1:4]) - reference_point

                # Map hand position to robot target coordinates
                X = 7 - displacement[2]/150
                Y = -displacement[0]/200 
                Z = 5 + displacement[1]/75 #works good

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

                    print('\nCalculated Joint Angles in Degrees:')
                    print(f't1: {np.degrees(angles[0]):.2f}°')
                    print(f't2: {np.degrees(angles[1]):.2f}°')
                    print(f't3: {np.degrees(angles[2]):.2f}°')

                    # Send angles to servos
                    asyncio.run(send_angles(angles, handIsClosed))

                except ValueError as e:
                    print(f"Inverse kinematics error: {e}")

            # Display hand tracking info
            cv2.putText(frame, ("Hand Position: " + str(centerOfMassNoFingers[1:])), (5,60), cv2.FONT_HERSHEY_PLAIN, fontSize, (0,255,0), fontThickness)

        else:
            cv2.putText(frame, ("Awaiting Hand..."), (5,70), cv2.FONT_HERSHEY_PLAIN, 2, (74,26,255), 2)

        frame_count += 1

        if cv2.waitKey(1) == ord('x'):
            break

        cv2.imshow('Hand Movement Interpreter', frame)

if __name__ == "__main__":
    main()