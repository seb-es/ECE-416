#!/usr/bin/env python3
import numpy as np
import keyboard
import time
import asyncio
import websockets
import json

# Set the websocket server URI (change to your Raspberry Pi's IP address and port)
SERVER_URI = "ws://10.203.94.66:8765"

def solve_angles(X, Y, Z, X0, Y0, Z0, a2, a3):
    """
    Solves inverse kinematics for a 3-DOF robotic arm.
    - Calculates angles **in radians**.
    - Returns **radian values**.
    - Enforces angle constraints:
      t1: [-90°, 90°]
      t2: [0°, 180°]
      t3: [-150°, 30°]
    """
    # Compute t1 (base rotation)
    t1 = np.arctan2(Y, X)
    
    # Check t1 constraint (-90° to 90°)
    if not (-np.pi/2 <= t1 <= np.pi/2):
        raise ValueError("t1 angle outside valid range [-90°, 90°]")

    # Compute D
    numerator = (X - X0)**2 + (Y - Y0)**2 + (Z - Z0)**2 - a2**2 - a3**2
    denominator = 2 * a2 * a3
    D = numerator / denominator

    # Ensure D is within valid range for acos
    if abs(D) > 1:
        raise ValueError("Target position is unreachable")

    # Compute both possible t3 solutions (elbow-up and elbow-down)
    t3_up = np.arctan2(np.sqrt(1 - D**2), D)
    t3_down = np.arctan2(-np.sqrt(1 - D**2), D)

    # Function to compute t2 for a given t3
    def compute_t2(t3):
        phi1 = np.arctan2(Z - Z0, np.sqrt((X - X0)**2 + (Y - Y0)**2))
        phi2 = np.arctan2(a3 * np.sin(t3), a2 + a3 * np.cos(t3))
        return phi1 - phi2

    # Compute t2 for both configurations
    t2_up = compute_t2(t3_up)
    t2_down = compute_t2(t3_down)

    # Check angle constraints and find valid solution
    solutions = []
    
    # Check elbow-up solution
    if (0 <= t2_up <= np.pi and  # t2: [0°, 180°]
        -5*np.pi/6 <= t3_up <= np.pi/6):  # t3: [-150°, 30°]
        solutions.append((t1, t2_up, t3_up))
        
    # Check elbow-down solution
    if (0 <= t2_down <= np.pi and  # t2: [0°, 180°]
        -5*np.pi/6 <= t3_down <= np.pi/6):  # t3: [-150°, 30°]
        solutions.append((t1, t2_down, t3_down))

    if not solutions:
        raise ValueError("No solutions found within angle constraints")
        
    return solutions[0]  # Return first valid solution

async def send_angles(t1, t2, t3):
    """
    Converts radian angles to degrees and applies servo range adjustments before sending.
    """
    try:
        async with websockets.connect(SERVER_URI) as websocket:
            # Convert radians to degrees and apply offsets
            servo_1 = np.degrees(t1) + 90
            servo_2 = np.degrees(t2) 
            servo_3 = 180-(np.degrees(t3)+150)  # Flip if needed

            # Create array of 6 angles - first 3 calculated, last 3 fixed at 90
            servo_angles = [int(servo_1), int(servo_2), int(servo_3), 90, 90, 90]
            
            # Create payload and send
            data = {"servo_angles": servo_angles}
            message = json.dumps(data)
            await websocket.send(message)
            print(f"\nSent angles to servos: {message}")
            
    except websockets.exceptions.ConnectionClosed:
        print("\nConnection closed. Reconnecting...")
    except Exception as e:
        print(f"\nError occurred: {e}")

async def main():
    """
    Runs the main control loop, allowing real-time movement of the robotic arm.
    """
    # Initial coordinates and constants
    X, Y, Z = 10, 0, 3.75 # Target coordinates
    X0, Y0, Z0 = 0.5, 0, 3.75  # Base position
    a2, a3 = 4.75, 7.5  # Link lengths
    step = 0.25  # Step size for coordinate changes

    while True:
        try:
            # Check for key presses and update coordinates
            if keyboard.is_pressed('z'):
                X += step
            elif keyboard.is_pressed('x'):
                X -= step
            elif keyboard.is_pressed('c'):
                Y += step
            elif keyboard.is_pressed('v'):
                Y -= step
            elif keyboard.is_pressed('b'):
                Z += step
            elif keyboard.is_pressed('n'):
                Z -= step
                
            # Calculate angles **in radians**
            t1, t2, t3 = solve_angles(X, Y, Z, X0, Y0, Z0, a2, a3)

            # Display calculated angles (converted to degrees for readability)
            print(f"\rCoordinates (X,Y,Z): ({X:.2f}, {Y:.2f}, {Z:.2f}) | "
                  f"Angles (t1,t2,t3): ({np.degrees(t1):.2f}°, {np.degrees(t2):.2f}°, {np.degrees(t3):.2f}°)", end='')

            # Send angles (converted properly) to servos
            await send_angles(t1, t2, t3)
            
            await asyncio.sleep(0.1)  # Small delay to prevent excessive calculations
            
        except ValueError as e:
            print(f"\rError: {e}", end='')
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram stopped.")
