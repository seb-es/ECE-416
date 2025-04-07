#!/usr/bin/env python3
import asyncio
import websockets
import json

# Set the websocket server URI (change to your Raspberry Pi's IP address and port)
SERVER_URI = "ws://10.203.94.66:8765"

async def send_servo_angles():
    while True:
        try:
            # Connect to the websocket server hosted on your Raspberry Pi
            async with websockets.connect(SERVER_URI) as websocket:
                # Get user input for 6 angles
                try:
                    input_str = input("Enter 6 angles (0 to 180) separated by commas: ")
                    servo_angles = [int(x.strip()) for x in input_str.split(',')]
                    
                    # Validate input
                   # if len(servo_angles) != 6:
                   #     print("Please enter exactly 6 values")
                   #    continue
                        
                    if not all(0 <= angle <= 180 for angle in servo_angles):
                        print("All angles must be between 0 and 180 degrees")
                        continue
                    
                    # Create a payload (as a dictionary) and then convert it to JSON
                    data = {"servo_angles": servo_angles}
                    message = json.dumps(data)
                    
                    # Send the JSON message to the server
                    await websocket.send(message)
                    print(f"Sent: {message}")
                    
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
