#!/usr/bin/env python3
import asyncio
import websockets
import json
import time

# Replace with your server's IP address and port
SERVER_URI = "ws://10.203.94.66:8765"

async def measure_latency():
    latency_values = []
    for _ in range(100):
        try:
            async with websockets.connect(SERVER_URI) as websocket:
                # Record the high-precision send time
                t_send = time.perf_counter()
                
                # Send timestamp to server
                message = json.dumps({"timestamp": t_send})
                await websocket.send(message)
                print(f"Sent: {message}")

                # Await response with echoed timestamp
                response = await asyncio.wait_for(websocket.recv(), timeout=2)
                t_receive = time.perf_counter()

                # Parse the response
                try:
                    data = json.loads(response)
                    t_echo = data.get("timestamp", None)
                    
                    if t_echo is not None:
                        rtt = (t_receive - t_send) * 1000  # Round-trip time in ms
                        latency_values.append(rtt)
                        print(f"Latency (RTT): {rtt:.2f} ms\n")
                    else:
                        print("Invalid response format: missing timestamp\n")

                except json.JSONDecodeError:
                    print("Received invalid JSON\n")
                
                await asyncio.sleep(1)

        except websockets.exceptions.ConnectionClosed:
            print("Connection closed. Reconnecting...\n")
            await asyncio.sleep(1)
        except asyncio.TimeoutError:
            print("Timeout waiting for server response\n")
        except Exception as e:
            print(f"Error occurred: {e}\n")
            await asyncio.sleep(1)
    
    average_latency = sum(latency_values) / len(latency_values)
    print(f"Average Latency: {average_latency:.2f} ms")

if __name__ == "__main__":
    try:
        asyncio.get_event_loop().run_until_complete(measure_latency())
    except KeyboardInterrupt:
        print("Client stopped.")
