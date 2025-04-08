import os
import sys
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort

# Print CUDA information
print(f"OpenCV version: {cv2.__version__}")
print(f"CUDA available: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print(f"CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
    for i in range(cv2.cuda.getCudaEnabledDeviceCount()):
        print(f"CUDA device {i}: {cv2.cuda.getDevice()}")

# Print ONNX Runtime information
print(f"ONNXRuntime version: {ort.__version__}")
print(f"Available providers: {ort.get_available_providers()}")
print(f"Current device: {ort.get_device()}")

# Adjust N for better video feedback:
# - A higher N at high resolution improves speedy playback significantly.
# - A higher N at low resolution has minimal impact.


# Function to apply artistic style to close/far objects ensuring high resolution
def high_apply_artsyle(frame, h, w, style_transfer_model):
    # Start timer for performance measurement
    start_time = cv2.getTickCount()

    inp = cv2.dnn.blobFromImage(
        frame, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False
    )
    style_transfer_model.setInput(inp)
    stylized_output = style_transfer_model.forward()
    stylized_output = stylized_output.reshape(3, h, w)
    stylized_output[0] += 103.939
    stylized_output[1] += 116.779
    stylized_output[2] += 123.680
    stylized_output = stylized_output.transpose(1, 2, 0)
    stylized_output = np.clip(stylized_output, 0, 255).astype(np.uint8)

    # End timer and print performance info
    end_time = cv2.getTickCount()
    process_time = (end_time - start_time) / cv2.getTickFrequency() * 1000
    print(f"High-res style transfer time: {process_time:.2f}ms")

    return stylized_output


# Function to apply artistic style to close/far objects and resizes frame for faster computation
def low_apply_artsyle(frame, h, w, style_transfer_model):
    # Start timer for performance measurement
    start_time = cv2.getTickCount()

    small_frame = cv2.resize(frame, (w // 2, h // 2))
    inp = cv2.dnn.blobFromImage(
        small_frame,
        1.0,
        (w // 2, h // 2),
        (103.939, 116.779, 123.680),
        swapRB=False,
        crop=False,
    )
    style_transfer_model.setInput(inp)
    stylized_output = style_transfer_model.forward()
    stylized_output = stylized_output.reshape(3, h // 2, w // 2)
    stylized_output[0] += 103.939
    stylized_output[1] += 116.779
    stylized_output[2] += 123.680
    stylized_output = stylized_output.transpose(1, 2, 0)
    stylized_output = np.clip(stylized_output, 0, 255).astype(np.uint8)
    result = cv2.resize(stylized_output, (w, h))

    # End timer and print performance info
    end_time = cv2.getTickCount()
    process_time = (end_time - start_time) / cv2.getTickFrequency() * 1000
    print(f"Low-res style transfer time: {process_time:.2f}ms")

    return result


# Function to handle depth map extraction
def get_depth_map(frame, depth_session, h, w):
    # Start timer for performance measurement
    start_time = cv2.getTickCount()

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

    # End timer and print performance info
    end_time = cv2.getTickCount()
    process_time = (end_time - start_time) / cv2.getTickFrequency() * 1000
    print(f"Depth map extraction time: {process_time:.2f}ms")

    return depth_map


# The pre-trained style models and depth model
style_model_url_1 = "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/mosaic.t7"
style_model_url_2 = "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/udnie.t7"
style_model_path_1 = "mosaic.t7"
style_model_path_2 = "udnie.t7"

depth_model_url = "https://huggingface.co/julienkay/sentis-MiDaS/blob/main/onnx/midas_v21_small_256.onnx"  # download path doesn't work anymore
depth_model_path = "midas.onnx"


def process_frame(
    frame,
    depth_session,
    style_transfer_model_1,
    style_transfer_model_2,
    foreground,
    background,
    depth_only=False,  # Add new parameter
):
    """Process a single frame without temporal smoothing"""
    # Start timer for performance measurement
    start_time = cv2.getTickCount()

    h, w, _ = frame.shape
    depth_map = get_depth_map(frame, depth_session, h, w)

    if depth_only:
        end_time = cv2.getTickCount()
        total_time = (end_time - start_time) / cv2.getTickFrequency() * 1000
        print(f"Depth map extraction time: {total_time:.2f}ms")
        return depth_map

    close_mask = depth_map >= 127
    far_mask = depth_map < 127

    if foreground == "high":
        stylized_output_close = high_apply_artsyle(frame, h, w, style_transfer_model_1)
    else:
        stylized_output_close = low_apply_artsyle(frame, h, w, style_transfer_model_1)

    if background == "high":
        stylized_output_far = high_apply_artsyle(frame, h, w, style_transfer_model_2)
    else:
        stylized_output_far = low_apply_artsyle(frame, h, w, style_transfer_model_2)

    final_output = np.zeros_like(frame)
    final_output[close_mask] = stylized_output_close[close_mask]
    final_output[far_mask] = stylized_output_far[far_mask]

    # End timer and print performance info
    end_time = cv2.getTickCount()
    total_time = (end_time - start_time) / cv2.getTickFrequency() * 1000
    print(f"Total frame processing time: {total_time:.2f}ms")

    return final_output


def get_models(providers=None):
    """Initialize and return the models"""
    # Download NST models if not already present
    for url, path in [
        (style_model_url_1, style_model_path_1),
        (style_model_url_2, style_model_path_2),
    ]:
        if not os.path.exists(path):
            print(f"Downloading {path}...")
            urllib.request.urlretrieve(url, path)

    # Load NST models using OpenCV
    print("Loading style transfer models...")
    style_transfer_model_1 = cv2.dnn.readNetFromTorch(style_model_path_2)
    style_transfer_model_2 = cv2.dnn.readNetFromTorch(style_model_path_1)

    # GPU optimization options for NVDIA GPU and AMD RADEON GPU
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("CUDA is available! Targeting GPU...")
        style_transfer_model_1.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        style_transfer_model_1.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Model 1 configured to use CUDA backend and target")

        style_transfer_model_2.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        style_transfer_model_2.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Model 2 configured to use CUDA backend and target")

        # Check available backends in this OpenCV version
        try:
            cuda_backend = (
                style_transfer_model_1.getPreferableBackend()
                == cv2.dnn.DNN_BACKEND_CUDA
            )
            cuda_target = (
                style_transfer_model_1.getPreferableTarget() == cv2.dnn.DNN_TARGET_CUDA
            )
            print(f"Using CUDA backend: {cuda_backend}, CUDA target: {cuda_target}")
        except AttributeError:
            print(
                "Note: This OpenCV version doesn't support getPreferableBackend(), but CUDA is configured"
            )

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    elif ort.get_device() == "ROCM":
        print("CUDA not available, but ROCm (AMD GPU) is available.")
        if providers is None:
            providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]

    else:
        print("No GPU acceleration available. Running on CPU.")
        if providers is None:
            providers = ["CPUExecutionProvider"]

    # Load the MiDaS depth estimation model
    print(f"Loading depth model with providers: {providers}")
    try:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_profiling = False

        depth_session = ort.InferenceSession(
            depth_model_path, options, providers=providers
        )
        print(f"Depth model providers: {depth_session.get_providers()}")
        print(f"Depth model actual provider: {depth_session._providers}")
    except Exception as e:
        print(f"Error initializing ONNX Runtime with GPU providers: {e}")
        print("Falling back to CPU execution...")
        depth_session = ort.InferenceSession(
            depth_model_path, providers=["CPUExecutionProvider"]
        )

    # Run a test inference to warm up models and check performance
    print("Warming up models with test inference...")
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = get_depth_map(test_frame, depth_session, 480, 640)
    _ = high_apply_artsyle(test_frame, 480, 640, style_transfer_model_1)

    return style_transfer_model_1, style_transfer_model_2, depth_session


# Function to generate video feedback using user specified resolution and fps
def generate(N, foreground, background, camera_index=0, depth_only=False):
    style_transfer_model_1, style_transfer_model_2, depth_session = get_models()

    # Try to access webcam with the provided index
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Explicitly use DirectShow

    if not cap.isOpened():
        print(f"Could not open webcam with index {camera_index}.")
        # Try a few alternative indices
        for alt_index in [0, 1, 2]:
            if alt_index == camera_index:
                continue
            print(f"Trying camera index {alt_index}...")
            cap = cv2.VideoCapture(alt_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                print(f"Successfully opened camera with index {alt_index}")
                break

    if not cap.isOpened():
        print("Error: Could not open any webcam.")
        return

    # Check if camera is actually providing frames
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print(
            "Error: Camera opened but not providing frames. Try another camera index."
        )
        cap.release()
        return

    # Get webcam properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Webcam resolution: {frame_width}x{frame_height}, FPS: {fps}")

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

        start_time = cv2.getTickCount()
        processed = process_frame(
            frame,
            depth_session,
            style_transfer_model_1,
            style_transfer_model_2,
            foreground,
            background,
            depth_only=depth_only,
        )
        end_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (end_time - start_time)
        print(f"Frame processing time: {1000 / fps:.2f}ms")

        prev_frame = frame.copy()
        prev_stylized = processed.copy()

        # Display side by side
        combined = np.hstack((frame, processed))
        cv2.imshow("Original vs Processed", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press "q" to exit video
            break

    cap.release()
    cv2.destroyAllWindows()


# Function to process a single image/frame
def process_image(
    frame,
    depth_session,
    style_transfer_model_1,
    style_transfer_model_2,
    foreground,
    background,
    prev_frame=None,
    prev_stylized=None,
):
    h, w, _ = frame.shape

    # Get the depth map for the frame
    depth_map = get_depth_map(frame, depth_session, h, w)

    # Create masks for close and far regions
    close_mask = depth_map >= 127  # Closer objects have higher depth values
    far_mask = depth_map < 127  # Farther objects have lower depth values

    # Apply style transfer for close objects using 1st style model
    if foreground == "high":
        stylized_output_close = high_apply_artsyle(frame, h, w, style_transfer_model_1)
    else:
        stylized_output_close = low_apply_artsyle(frame, h, w, style_transfer_model_1)

    # Apply style transfer for far objects using 2nd style model
    if background == "high":
        stylized_output_far = high_apply_artsyle(frame, h, w, style_transfer_model_2)
    else:
        stylized_output_far = low_apply_artsyle(frame, h, w, style_transfer_model_2)

    # Blend the results based on depth regions
    final_output = np.zeros_like(frame)
    final_output[close_mask] = stylized_output_close[close_mask]
    final_output[far_mask] = stylized_output_far[far_mask]

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

    # Update prev_stylized reference (for the calling function)
    if prev_stylized is not None:
        prev_stylized[:] = final_output

    # Display the final output
    cv2.imshow("Artistic Depth Feedback", final_output)
    return final_output


if __name__ == "__main__":
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Artistic Style Transfer with Depth")
    parser.add_argument(
        "N",
        type=int,
        nargs="?",
        default=5,
        help="Frame skipping factor (higher = faster but less smooth)",
    )
    parser.add_argument(
        "foreground",
        nargs="?",
        choices=["high", "low"],
        default="high",
        help="Quality for foreground style transfer",
    )
    parser.add_argument(
        "background",
        nargs="?",
        choices=["high", "low"],
        default="low",
        help="Quality for background style transfer",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use")
    parser.add_argument(
        "--depth-only", action="store_true", help="Show only depth map visualization"
    )

    args = parser.parse_args()

    # Run with parsed arguments
    generate(args.N, args.foreground, args.background, args.camera, args.depth_only)
