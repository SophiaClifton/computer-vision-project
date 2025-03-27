import os
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort

# NOTES: COMPUTATIONALLY EXPENSIVE, test.py is wip of optimized version

# Temporal smoothing blending parameter (adjust to control the influence of previous frames)
lambda_temp = 0.3


def apply_artsyle_close(frame, h, w, style_transfer_model):
    # Convert frame to blob for OpenCV's dlm
    # Mean subtraction normalizes input using ImageNet-trained model values
    inp_close = cv2.dnn.blobFromImage(
        frame, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False
    )
    # Feed frame into style transfer model # 1
    style_transfer_model.setInput(inp_close)
    stylized_output_close = (
        style_transfer_model.forward()
    )  # Generates styled version by forward propagating in neural net
    stylized_output_close = stylized_output_close.reshape(
        3, h, w
    )  # Brings back RGB format

    # Re-add mean values to restore color distribution
    stylized_output_close[0] += 103.939
    stylized_output_close[1] += 116.779
    stylized_output_close[2] += 123.680

    # Transpose output to match OpenCV's expected format
    stylized_output_close = stylized_output_close.transpose(1, 2, 0)
    stylized_output_close = np.clip(stylized_output_close, 0, 255).astype(np.uint8)
    return stylized_output_close


def apply_artsyle_far(frame, h, w, style_transfer_model):
    # Convert frame to blob for OpenCV's dlm
    # Mean subtraction normalizes input using ImageNet-trained model values
    inp_far = cv2.dnn.blobFromImage(
        frame, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False
    )  # Converts frame into blob, needed for open cvs dlm

    # Feed frame into style transfer model # 2
    style_transfer_model.setInput(inp_far)
    stylized_output_far = (
        style_transfer_model.forward()
    )  # Generates styled version by forward propagating in neural net
    stylized_output_far = stylized_output_far.reshape(3, h, w)  # Brings back RGB format

    # Re-add mean values to restore color distribution
    stylized_output_far[0] += 103.939
    stylized_output_far[1] += 116.779
    stylized_output_far[2] += 123.680

    # Transpose output to match OpenCV's expected format
    stylized_output_far = stylized_output_far.transpose(1, 2, 0)
    stylized_output_far = np.clip(stylized_output_far, 0, 255).astype(np.uint8)
    return stylized_output_far


def get_depth_map(frame, depth_session, h, w):
    # Resize frame to 256x256 for MiDaS input
    depth_input = cv2.resize(frame, (256, 256))
    # Normalize the image to [0, 1]-> MiDaS will complain otherwise
    depth_input = depth_input.astype(np.float32) / 255.0
    # Convert image to NCHW format (batch_size, channels, height, width)
    depth_input = np.transpose(depth_input, (2, 0, 1))  # (channels, height, width)
    depth_input = np.expand_dims(depth_input, axis=0)  # Add batch dimension

    # Run depth estimation using onnxruntime
    depth_map = depth_session.run(None, {"input_image": depth_input})[0]

    # Rescale depth map to original image size
    depth_map = cv2.resize(depth_map[0], (w, h))

    # Normalize depth map to 0-255 range
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return depth_map


# The pre-trained style models
style_model_url_1 = "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/mosaic.t7"
style_model_url_2 = "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/udnie.t7"
style_model_path_1 = "mosaic.t7"
style_model_path_2 = "udnie.t7"

# MiDaS model for depth estimation
depth_model_url = "https://huggingface.co/julienkay/sentis-MiDaS/blob/main/onnx/midas_v21_small_256.onnx"
depth_model_path = "midas.onnx"

# Download models if not already
for url, path in [
    (style_model_url_1, style_model_path_1),
    (style_model_url_2, style_model_path_2),
    (depth_model_url, depth_model_path),
]:
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        urllib.request.urlretrieve(url, path)
        print(f"{path} downloaded successfully.")

# Load the style transfer models using OpenCV
style_transfer_model_1 = cv2.dnn.readNetFromTorch(style_model_path_1)
style_transfer_model_2 = cv2.dnn.readNetFromTorch(style_model_path_2)

# Load the MiDaS depth estimation model using onnxruntime
depth_session = ort.InferenceSession(depth_model_path)

# Access webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize previous frame variables for temporal smoothing
prev_frame = None
prev_stylized = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame.")
        break

    h, w, _ = frame.shape

    # Get the depth map for the frame
    depth_map = get_depth_map(frame, depth_session, h, w)

    # Apply colormap -> red for close and blue for far
    # depth_colormap = cv2.applyColorMap(255 - depth_map, cv2.COLORMAP_JET)

    # Create masks for close and far regions
    close_mask = depth_map < 127  # Since closer objects have lower depth values
    far_mask = depth_map >= 127  # Since farther objects have higher depth values

    # Apply style transfer for close objects -> 1st style model
    stylized_output_close = apply_artsyle_close(frame, h, w, style_transfer_model_1)

    # Apply style transfer for far objects -> 2nd style model
    stylized_output_far = apply_artsyle_far(frame, h, w, style_transfer_model_2)

    # Blend the results based on depth regions
    final_output = np.zeros_like(frame)
    final_output[close_mask] = stylized_output_close[
        close_mask
    ]  # Only apply close-artsyle on close mask region
    final_output[far_mask] = stylized_output_far[
        far_mask
    ]  # Only apply far-artsyle on far mask region

    # Temporal smoothing block using optical flow
    if prev_frame is not None and prev_stylized is not None:
        # Convert previous and current frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate optical flow between previous and current frame
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        # Warp previous stylized frame using computed optical flow
        h_frame, w_frame = final_output.shape[:2]
        flow_map = -flow  # Reverse flow for warping
        grid_x, grid_y = np.meshgrid(np.arange(w_frame), np.arange(h_frame))
        remap_x = (grid_x + flow_map[..., 0]).astype(np.float32)
        remap_y = (grid_y + flow_map[..., 1]).astype(np.float32)
        warped_prev = cv2.remap(
            prev_stylized, remap_x, remap_y, interpolation=cv2.INTER_LINEAR
        )

        # Blend current output with warped previous output for temporal smoothing
        final_output = cv2.addWeighted(
            final_output, 1 - lambda_temp, warped_prev, lambda_temp, 0
        )

    # Display the final output
    cv2.imshow("Artistic Depth Feedback", final_output)

    # Update previous frame variables
    prev_frame = frame.copy()
    prev_stylized = final_output.copy()

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press "q" to exit video
        break

cap.release()
cv2.destroyAllWindows()
