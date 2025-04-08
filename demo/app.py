import base64
import json
import logging

import cv2
import numpy as np
from flask import Flask
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
from werkzeug.middleware.dispatcher import DispatcherMiddleware

import demo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=".", static_url_path="")

# Initialize models once at startup
style_model_1, style_model_2, depth_session = demo.get_models()

# Global variables for streaming
connected_clients = set()
prev_frames = {}  # Store previous frames for each client
prev_stylized = {}  # Store previous stylized frames for each client


@app.route("/")
def index():
    return app.send_static_file("demo.html")


def ws_app(environ, start_response):
    """WebSocket handler function"""
    if environ.get("HTTP_UPGRADE", "").lower() != "websocket":
        # Return 400 Bad Request if this is not a WebSocket request
        start_response("400 Bad Request", [("Content-Type", "text/plain")])
        return [b"Expected WebSocket request"]

    ws = environ["wsgi.websocket"]
    if not ws:
        start_response("400 Bad Request", [("Content-Type", "text/plain")])
        return [b"Expected WebSocket connection"]

    try:
        connected_clients.add(ws)
        logger.info("Client connected")

        while True:
            try:
                message = ws.receive()
                if message is None:
                    break

                # Parse the message
                data = json.loads(message)
                if data.get("type") == "start":
                    ws.send(json.dumps({"type": "started"}))
                    continue
                elif data.get("type") == "stop":
                    ws.send(json.dumps({"type": "stopped"}))
                    continue

                # Handle frame data
                encoded_data = data["image"].split(",")[1]
                depth_only = data.get("depth_only", False)

                # Decode image
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                # Process frame
                client_id = id(ws)
                processed = demo.process_image(
                    frame,
                    depth_session,
                    style_model_1,
                    style_model_2,
                    "high",
                    "high",
                    prev_frames.get(client_id),
                    prev_stylized.get(client_id),
                )

                # Store frames for next iteration
                if not depth_only:
                    prev_frames[client_id] = frame.copy()
                    prev_stylized[client_id] = processed.copy()

                # Convert processed frame to base64
                _, buffer = cv2.imencode(".jpg", processed)
                processed_b64 = base64.b64encode(buffer).decode("utf-8")

                # Send processed frame back to client
                ws.send(
                    json.dumps(
                        {
                            "type": "frame",
                            "data": f"data:image/jpeg;base64,{processed_b64}",
                        }
                    )
                )

            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                continue

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connected_clients.remove(ws)
        logger.info("Client disconnected")

    return []


# Create dispatcher middleware to handle both HTTP and WebSocket
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {"/ws": ws_app})

if __name__ == "__main__":
    # Run the server with WebSocket support
    server = pywsgi.WSGIServer(("0.0.0.0", 5000), app, handler_class=WebSocketHandler)
    print("Server is running on http://localhost:5000")
    server.serve_forever()
