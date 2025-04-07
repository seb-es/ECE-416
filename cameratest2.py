import tkinter as tk
from PIL import Image, ImageTk
import websocket
import threading
import base64
import io

# ==== Global reference to image ====
current_image = None

def on_message(ws, message):
    global current_image

    try:
        if message.startswith("data:image/jpeg;base64,"):
            # Strip the prefix
            base64_data = message.split(",")[1]
            img_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(img_data))

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)

            # Update the GUI in the main thread
            def update_img():
                global current_image
                current_image = photo  # Prevent garbage collection
                image_label.config(image=photo)

            root.after(0, update_img)
    except Exception as e:
        print("Error handling frame:", e)

def on_error(ws, error):
    print("❌ WebSocket error:", error)

def on_close(ws, close_status_code, close_msg):
    print("🔌 WebSocket closed")

def on_open(ws):
    print("✅ WebSocket connected")

def start_websocket():
    ws = websocket.WebSocketApp(
        "ws://10.203.94.66:8766",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    ws.run_forever()

root = tk.Tk()
root.title("📷 Live Stream")
root.configure(bg='black')
root.geometry("800x600")

image_label = tk.Label(root, bg='black')
image_label.pack(expand=True)

# Start WebSocket in a separate thread
threading.Thread(target=start_websocket, daemon=True).start()

root.mainloop()


