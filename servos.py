#!/usr/bin/env python3
import asyncio
import json
import websockets
from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)

async def handler(websocket, path):
    print(f"Client connected: {websocket.remote_address}")
    try:
        message = await websocket.recv()
        try:
            data = json.loads(message)
            servo_angles = data.get("servo_angles", [])
            print(f"Received servo angles: {servo_angles}")
            
            if len(servo_angles) == 3:
                for i in range(3):
                    kit.servo[i].angle = int(servo_angles[i])
            else:
                print("Error: Expected exactly 3 servo angles")
                
        except json.JSONDecodeError:
            print("Error: Received a non-JSON message.")
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")

async def main():
    async with websockets.serve(handler, "10.203.94.66", 8765):
        print("WebSocket server is running on ws://10.203.94.66:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
