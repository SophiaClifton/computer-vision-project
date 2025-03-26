import cv2
import numpy as np
import urllib.request
import os
import onnxruntime as ort
import sys

# To optimize performance, divide h and w by //2 in the style transfer functions.
# This reduces computation time and frame resolution, resulting in lower quality but faster processing for real-time video playback.

# Adjust N for better video feedback:
# - A higher N at high resolution improves speedy playback significantly.
# - A higher N at low resolution has minimal impact.

# Function to apply artistic style to close objects
def high_apply_artsyle_close(frame, h, w, style_transfer_model):
    #small_frame = cv2.resize(frame, (w, h // 2))  # Resize for faster inference
    inp_close = cv2.dnn.blobFromImage(frame, 1.0, (w , h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    style_transfer_model.setInput(inp_close)
    stylized_output_close = style_transfer_model.forward()
    stylized_output_close = stylized_output_close.reshape(3, h, w)
    stylized_output_close[0] += 103.939
    stylized_output_close[1] += 116.779
    stylized_output_close[2] += 123.680
    stylized_output_close = stylized_output_close.transpose(1, 2, 0)
    stylized_output_close = np.clip(stylized_output_close, 0, 255).astype(np.uint8)
    #return cv2.resize(stylized_output_close, (w, h)) use this instead if using //2 for lower res
    return stylized_output_close

# Function to apply artistic style to close objects
def low_apply_artsyle_close(frame, h, w, style_transfer_model):
    small_frame = cv2.resize(frame, (w, h // 2))  # Resize for faster inference
    inp_close = cv2.dnn.blobFromImage(small_frame, 1.0, (w //2, h//2), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    style_transfer_model.setInput(inp_close)
    stylized_output_close = style_transfer_model.forward()
    stylized_output_close = stylized_output_close.reshape(3, h//2, w//2)
    stylized_output_close[0] += 103.939
    stylized_output_close[1] += 116.779
    stylized_output_close[2] += 123.680
    stylized_output_close = stylized_output_close.transpose(1, 2, 0)
    stylized_output_close = np.clip(stylized_output_close, 0, 255).astype(np.uint8)
    return cv2.resize(stylized_output_close, (w, h)) 

# Function to apply artistic style to far objects
def high_apply_artsyle_far(frame, h, w, style_transfer_model):
    inp_far = cv2.dnn.blobFromImage(frame, 1.0, (w, h ), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    style_transfer_model.setInput(inp_far)
    stylized_output_far = style_transfer_model.forward()
    stylized_output_far = stylized_output_far.reshape(3, h, w)
    stylized_output_far[0] += 103.939
    stylized_output_far[1] += 116.779
    stylized_output_far[2] += 123.680
    stylized_output_far = stylized_output_far.transpose(1, 2, 0)
    stylized_output_far = np.clip(stylized_output_far, 0, 255).astype(np.uint8)
    return stylized_output_far

# Function to apply artistic style to far objects
def low_apply_artsyle_far(frame, h, w, style_transfer_model):
    small_frame = cv2.resize(frame, (w // 2, h // 2))  # Resize for faster inference
    inp_far = cv2.dnn.blobFromImage(small_frame, 1.0, (w//2, h//2 ), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    style_transfer_model.setInput(inp_far)
    stylized_output_far = style_transfer_model.forward()
    stylized_output_far = stylized_output_far.reshape(3, h//2, w//2)
    stylized_output_far[0] += 103.939
    stylized_output_far[1] += 116.779
    stylized_output_far[2] += 123.680
    stylized_output_far = stylized_output_far.transpose(1, 2, 0)
    stylized_output_far = np.clip(stylized_output_far, 0, 255).astype(np.uint8)
    return cv2.resize(stylized_output_far, (w, h))

# Function to handle depth map extraction
def get_depth_map(frame, depth_session, h, w):
    # Resize for depth estimation (lower resolution)
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

depth_model_url = "https://huggingface.co/julienkay/sentis-MiDaS/blob/main/onnx/midas_v21_small_256.onnx"
depth_model_path = "midas.onnx"

if len(sys.argv) != 4:
    print("Usage: script.py <n: 1-5> <foreground: high/low> <background: high/low>")
    sys.exit(1)

N = int(sys.argv[1])
foreground = sys.argv[2].lower()
background = sys.argv[3].lower()
    
# Download models if not already present
for url, path in [(style_model_url_1, style_model_path_1), (style_model_url_2, style_model_path_2), (depth_model_url, depth_model_path)]:
    if not os.path.exists(path):
        #print(f"Downloading {path}...")
        urllib.request.urlretrieve(url, path)
        #print(f"{path} downloaded successfully.")

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

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

elif ort.get_device() == "ROCM":  # ROCm support for AMD GPUs (Linux)
    print("CUDA not available, but ROCm (AMD GPU) is available.")
    providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']

else:
    print("No GPU acceleration available. Running on CPU.")
    providers = ['CPUExecutionProvider']

# Load the MiDaS depth estimation model using ONNX Runtime
depth_session = ort.InferenceSession(depth_model_path, providers=providers)

# Access webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_count = 0
# N = 5  # Apply every N frames, N=2 slower but better fps, N=5 choppier but real-time -> can only tell diff at higher res img
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
    if(foreground == "high"):
        stylized_output_close = high_apply_artsyle_close(frame, h, w, style_transfer_model_1)
    else:
        stylized_output_close = low_apply_artsyle_close(frame, h, w, style_transfer_model_1)


    # Apply style transfer for far objects -> 2nd style model
    if(background == "high"):
        stylized_output_far = high_apply_artsyle_far(frame, h, w, style_transfer_model_2)
    else:
        stylized_output_far = low_apply_artsyle_far(frame, h, w, style_transfer_model_2)

    # Blend the results based on depth regions
    final_output = np.zeros_like(frame)
    final_output[close_mask] = stylized_output_close[close_mask]  # Close objects
    final_output[far_mask] = stylized_output_far[far_mask]  # Far objects
    
    # Display the final output
    cv2.imshow("Artistic Depth Feedback", final_output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press "q" to exit video
        break

cap.release()
cv2.destroyAllWindows()

