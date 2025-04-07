import asyncio
import cv2
import websockets
import base64

# Open the webcam (0 = default camera; change to 1 if needed)
camera = cv2.VideoCapture(0)

# Frame generator
def get_frames():
    while True:
        success, frame = camera.read()
        if not success:
            print("❌ Failed to read frame from webcam")
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        frame = base64.b64encode(buffer)
        # ✅ Return as UTF-8 string with proper data URL prefix
        yield (b'data:image/jpeg;base64,' + frame).decode('utf-8')

# WebSocket connection handler
async def handle_connection(websocket):
    print("✅ Client connected")
    try:
        for frame in get_frames():
            await websocket.send(frame)  # Send as string
            await asyncio.sleep(0.03)    # ~30 FPS
    except websockets.exceptions.ConnectionClosed:
        print("🔌 Client disconnected")
    except Exception as e:
        print(f"🔥 Error: {e}")

# Start the server
async def main():
    async with websockets.serve(handle_connection, "localhost", 8000):
        print("🚀 WebSocket server running at ws://localhost:8000")
        await asyncio.Future()  # Keeps the server alive

if __name__ == "__main__":
    asyncio.run(main())
