#!/usr/bin/env python3
import asyncio
import websockets
import json
import numpy as np
from three import *

# Set the websocket server URI (change to your Raspberry Pi's IP address and port)
SERVER_URI = "ws://10.203.94.66:8765"

async def send_servo_angles():
    while True:
        try:
            # Connect to the websocket server hosted on your Raspberry Pi
            # Initialize robotic arm parameters
            k = np.array([[0,0,1], [0,0,1], [0,0,1]])  # Joint rotation axes
    
            a1, a2, a3, a4, a5 = 1, 1.75, 1.5, 4.25, 7
            t = np.array([[0,0,0], [a2,0,a1], [0,a4,0]])  # Joint translations
    
            p_eff_2 = [7,a3+4,0]  # End effector position
            k_c = RoboticArm(k,t)
            q_0 = np.array([0,0,0])  # Starting angles
            
            async with websockets.connect(SERVER_URI) as websocket:
                try:
                    # Get user input for x,y,z coordinates
                    input_str = input("Enter x,y,z coordinates for end effector (separated by commas): ")
                    coords = [float(x.strip()) for x in input_str.split(',')]
                    
                    if len(coords) != 3:
                        print("Please enter exactly 3 coordinates (x,y,z)")
                        continue
                    
                    # Calculate joint angles using inverse kinematics
                    endeffector_goal_position = np.array(coords)
                    
                    # Check if goal position is within bounds
                    magnitude = np.sqrt(np.sum(np.square(endeffector_goal_position)))
                    if magnitude > 12:
                        print(f"Goal position magnitude {magnitude:.2f} exceeds maximum reach of 12 units")
                        print("Position is out of bounds")
                        continue
                        
                    final_q = k_c.pseudo_inverse(q_0, p_eff_N=p_eff_2, 
                                               goal_position=endeffector_goal_position, 
                                               max_steps=500)
                    
                    # Convert angles to degrees and add 90 degrees to first 3 angles
                    servo_angles = [int(np.degrees(angle) + 90) for angle in final_q]
                    
                    # Add fixed angles for remaining servos
                    servo_angles.extend([90, 90, 90])
                    
                    # Create payload and send
                    data = {"servo_angles": servo_angles}
                    message = json.dumps(data)
                    await websocket.send(message)
                    print(f"Sent angles to servos: {message}")
                    
                except ValueError:
                    print("Invalid input. Please enter numbers separated by commas")
                    continue
                    
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed. Reconnecting...")
            await asyncio.sleep(1)
        except Exception as e:
            print(f"Error occurred: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.get_event_loop().run_until_complete(send_servo_angles())
    except KeyboardInterrupt:
        print("Client stopped.")
