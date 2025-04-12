import queue
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

import demo

app = Flask(__name__, static_folder=".", static_url_path="")

# Initialize models once at startup
style_model_1, style_model_2, depth_session = demo.get_models()

# Global variables for streaming
frame_queue = queue.Queue(maxsize=10)
processed_frames = queue.Queue(maxsize=10)
streaming_active = False
processing_thread = None

# Store previous frames for temporal smoothing
prev_frames = {}
prev_stylized = {}


@app.route("/")
def index():
    return app.send_static_file("demo.html")


@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    if "frame" not in request.files:
        return "No frame data", 400

    # Read frame from request
    frame_file = request.files["frame"]
    frame_data = frame_file.read()

    # Convert to numpy array
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return "Invalid frame data", 400

    # Add frame to queue if streaming is active
    if streaming_active and not frame_queue.full():
        frame_queue.put(frame)

    return jsonify({"status": "success"})


def process_frame_with_depth_only(frame, depth_session):
    """Handle depth-only processing mode"""
    h, w, _ = frame.shape
    depth_map = demo.get_depth_map(frame, depth_session, h, w)
    return cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)


def process_frames_thread(depth_only=False):
    global streaming_active

    # Use thread ID as client ID for temporal smoothing
    client_id = threading.get_ident()

    while streaming_active:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Process the frame
            try:
                if depth_only:
                    processed = process_frame_with_depth_only(frame, depth_session)
                else:
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
                    # # Store frames for next iteration
                    prev_frames[client_id] = frame.copy()
                    prev_stylized[client_id] = processed.copy()

                if not processed_frames.full():
                    processed_frames.put(processed)
            except Exception as e:
                print(f"Error processing frame: {e}")

        time.sleep(0.01)  # Small sleep to prevent CPU overload


def generate_processed_frames():
    global streaming_active

    while streaming_active:
        if not processed_frames.empty():
            processed = processed_frames.get()

            # Encode frame as JPEG
            ret, buffer = cv2.imencode(".jpg", processed)
            if not ret:
                continue

            # Yield frame in multipart format
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        else:
            # If no frames, yield an empty part to keep connection alive
            time.sleep(0.03)


@app.route("/stream")
def stream():
    global streaming_active, processing_thread

    # Get depth_only parameter from query string
    depth_only = request.args.get("depth_only", "false").lower() == "true"
    streaming_active = True

    # Start processing thread if not already running
    if processing_thread is None or not processing_thread.is_alive():
        processing_thread = threading.Thread(
            target=process_frames_thread, args=(depth_only,)
        )
        processing_thread.daemon = True
        processing_thread.start()

    # Return a streaming response using multipart/x-mixed-replace
    return Response(
        generate_processed_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stop_stream", methods=["POST"])
def stop_stream():
    global streaming_active, processing_thread

    streaming_active = False

    # Clear the queues
    while not frame_queue.empty():
        frame_queue.get()
    while not processed_frames.empty():
        processed_frames.get()

    # Clean up temporal smoothing data
    if processing_thread:
        client_id = processing_thread.ident
        prev_frames.pop(client_id, None)
        prev_stylized.pop(client_id, None)

    # Wait for processing thread to end
    if processing_thread and processing_thread.is_alive():
        processing_thread.join(timeout=0)

    return jsonify({"status": "stopped"})


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
