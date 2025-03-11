#!/usr/bin/env python3
import asyncio
import websockets
import json
import random  # Replace with your own logic for generating servo angles

# Set the websocket server URI (change to your Raspberry Pi's IP address and port)
SERVER_URI = "ws://10.203.94.66:8765"

async def send_servo_angles():
    # Connect to the websocket server hosted on your Raspberry Pi
    async with websockets.connect(SERVER_URI) as websocket:
        while True:
            # For demonstration, generate 5 random angles between 0 and 180 degrees.
            # Replace this with your computation or sensor reading logic.
            servo_angles = [random.randint(-90, 90) for _ in range(5)]
            
            # Create a payload (as a dictionary) and then convert it to JSON
            data = {"servo_angles": servo_angles}
            message = json.dumps(data)
            
            # Send the JSON message to the server
            await websocket.send(message)
            print(f"Sent: {message}")
            
            # Update at a fixed interval (e.g., every 0.1 seconds)
            await asyncio.sleep(0.1)

if __name__ == "__main__":
    try:
        asyncio.get_event_loop().run_until_complete(send_servo_angles())
    except KeyboardInterrupt:
        print("Client stopped.")
