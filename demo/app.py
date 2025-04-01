import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

import demo

app = Flask(__name__, static_folder=".", static_url_path="")

# Initialize models once at startup
style_model_1, style_model_2, depth_session = demo.get_models()


@app.route("/")
def index():
    return app.send_static_file("demo.html")


@app.route("/process_frame", methods=["POST"])
def process_frame():
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

    # Process the frame
    processed = demo.process_frame(
        frame, depth_session, style_model_1, style_model_2, "high", "high"
    )

    # Encode processed frame
    _, buffer = cv2.imencode(".jpg", processed)
    return Response(buffer.tobytes(), mimetype="image/jpeg")


@app.route("/stop_stream", methods=["POST"])
def stop_stream():
    return jsonify({"status": "stopped"})


if __name__ == "__main__":
    app.run(debug=True)
