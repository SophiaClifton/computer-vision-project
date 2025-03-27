import os
import sys
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort

# To optimize performance, "low" resolution NST -> divide h and w by // 2

# Adjust N for better video feedback:
# - A higher N at high resolution improves speedy playback significantly.
# - A higher N at low resolution has minimal impact.


# Function to apply artistic style to close objects ensuring high resolution
def high_apply_artsyle_close(frame, h, w, style_transfer_model):
    inp_close = cv2.dnn.blobFromImage(
        frame, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False
    )
    style_transfer_model.setInput(inp_close)
    stylized_output_close = style_transfer_model.forward()
    stylized_output_close = stylized_output_close.reshape(3, h, w)
    stylized_output_close[0] += 103.939
    stylized_output_close[1] += 116.779
    stylized_output_close[2] += 123.680
    stylized_output_close = stylized_output_close.transpose(1, 2, 0)
    stylized_output_close = np.clip(stylized_output_close, 0, 255).astype(np.uint8)
    return stylized_output_close


# Function to apply artistic style to close objects opting for low resolution output
def low_apply_artsyle_close(frame, h, w, style_transfer_model):
    small_frame = cv2.resize(frame, (w, h // 2))  # Resize for faster inference
    inp_close = cv2.dnn.blobFromImage(
        small_frame,
        1.0,
        (w // 2, h // 2),
        (103.939, 116.779, 123.680),
        swapRB=False,
        crop=False,
    )
    style_transfer_model.setInput(inp_close)
    stylized_output_close = style_transfer_model.forward()
    stylized_output_close = stylized_output_close.reshape(3, h // 2, w // 2)
    stylized_output_close[0] += 103.939
    stylized_output_close[1] += 116.779
    stylized_output_close[2] += 123.680
    stylized_output_close = stylized_output_close.transpose(1, 2, 0)
    stylized_output_close = np.clip(stylized_output_close, 0, 255).astype(np.uint8)
    return cv2.resize(stylized_output_close, (w, h))


# Function to apply artistic style to far objects ensuring high resolution
def high_apply_artsyle_far(frame, h, w, style_transfer_model):
    inp_far = cv2.dnn.blobFromImage(
        frame, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False
    )
    style_transfer_model.setInput(inp_far)
    stylized_output_far = style_transfer_model.forward()
    stylized_output_far = stylized_output_far.reshape(3, h, w)
    stylized_output_far[0] += 103.939
    stylized_output_far[1] += 116.779
    stylized_output_far[2] += 123.680
    stylized_output_far = stylized_output_far.transpose(1, 2, 0)
    stylized_output_far = np.clip(stylized_output_far, 0, 255).astype(np.uint8)
    return stylized_output_far


# Function to apply artistic style to far objects opting for low resolution output
def low_apply_artsyle_far(frame, h, w, style_transfer_model):
    small_frame = cv2.resize(frame, (w // 2, h // 2))  # Resize for faster inference
    inp_far = cv2.dnn.blobFromImage(
        small_frame,
        1.0,
        (w // 2, h // 2),
        (103.939, 116.779, 123.680),
        swapRB=False,
        crop=False,
    )
    style_transfer_model.setInput(inp_far)
    stylized_output_far = style_transfer_model.forward()
    stylized_output_far = stylized_output_far.reshape(3, h // 2, w // 2)
    stylized_output_far[0] += 103.939
    stylized_output_far[1] += 116.779
    stylized_output_far[2] += 123.680
    stylized_output_far = stylized_output_far.transpose(1, 2, 0)
    stylized_output_far = np.clip(stylized_output_far, 0, 255).astype(np.uint8)
    return cv2.resize(stylized_output_far, (w, h))


# Function to handle depth map extraction
def get_depth_map(frame, depth_session, h, w):
    depth_input = cv2.resize(frame, (256, 256))
    depth_input = depth_input.astype(np.float32) / 255.0
    depth_input = np.transpose(depth_input, (2, 0, 1))  # Convert to NCHW format
    depth_input = np.expand_dims(depth_input, axis=0)  # Add batch dimension

    # Run depth estimation
    depth_map = depth_session.run(None, {"input_image": depth_input})[0]

    # Rescale depth map back to original resolution
    depth_map = cv2.resize(depth_map[0], (w, h))

    # Normalize depth map to 0-255 range
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return depth_map


# The pre-trained style models and depth model
style_model_url_1 = "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/mosaic.t7"
style_model_url_2 = "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/udnie.t7"
style_model_path_1 = "mosaic.t7"
style_model_path_2 = "udnie.t7"

depth_model_url = "https://huggingface.co/julienkay/sentis-MiDaS/blob/main/onnx/midas_v21_small_256.onnx"  # download path doesn't work anymore
depth_model_path = "midas.onnx"


# Main generate function
def generate(N, foreground, background):
    # Download NST models if not already present
    for url, path in [
        (style_model_url_1, style_model_path_1),
        (style_model_url_2, style_model_path_2),
    ]:
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)

    # Load the style transfer models using OpenCV
    style_transfer_model_1 = cv2.dnn.readNetFromTorch(style_model_path_2)
    style_transfer_model_2 = cv2.dnn.readNetFromTorch(style_model_path_1)

    # Check if CUDA is available and enable it if it is
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("CUDA is available! Running on GPU...")
        style_transfer_model_1.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        style_transfer_model_1.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        style_transfer_model_2.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        style_transfer_model_2.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    elif ort.get_device() == "ROCM":  # ROCm support for AMD GPUs (Linux)
        print("CUDA not available, but ROCm (AMD GPU) is available.")
        providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]

    else:
        print("No GPU acceleration available. Running on CPU.")
        providers = ["CPUExecutionProvider"]

    # Load the MiDaS depth estimation model using ONNX Runtime
    depth_session = ort.InferenceSession(depth_model_path, providers=providers)

    # Access webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    frame_count = 0

    # Initialize previous frame variables for temporal smoothing
    prev_frame = None
    prev_stylized = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame.")
            break

        frame_count += 1
        if frame_count % N != 0:  # Skip N-1 frames for performance
            continue

        h, w, _ = frame.shape

        # Get the depth map for the frame
        depth_map = get_depth_map(frame, depth_session, h, w)

        # Create masks for close and far regions
        close_mask = depth_map >= 127  # Closer objects have lower depth values
        far_mask = depth_map < 127  # Farther objects have higher depth values

        # Apply style transfer for close objects -> 1st style model
        if foreground == "high":
            stylized_output_close = high_apply_artsyle_close(
                frame, h, w, style_transfer_model_1
            )
        else:
            stylized_output_close = low_apply_artsyle_close(
                frame, h, w, style_transfer_model_1
            )

        # Apply style transfer for far objects -> 2nd style model
        if background == "high":
            stylized_output_far = high_apply_artsyle_far(
                frame, h, w, style_transfer_model_2
            )
        else:
            stylized_output_far = low_apply_artsyle_far(
                frame, h, w, style_transfer_model_2
            )

        # Blend the results based on depth regions
        final_output = np.zeros_like(frame)
        final_output[close_mask] = stylized_output_close[close_mask]  # Close objects
        final_output[far_mask] = stylized_output_far[far_mask]  # Far objects

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
            final_output = cv2.addWeighted(final_output, 1 - 0.3, warped_prev, 0.3, 0)

        # Display the final output
        cv2.imshow("Artistic Depth Feedback", final_output)

        prev_frame = frame.copy()
        prev_stylized = final_output.copy()

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press "q" to exit video
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    generate(int(sys.argv[1]), sys.argv[2], sys.argv[3])
