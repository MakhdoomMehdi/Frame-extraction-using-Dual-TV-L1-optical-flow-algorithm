import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from raft import RAFT
from raft.utils.utils import InputPadder

# Paths to input data and output storage
dataset_path = "D:/Computer Vision/Dataset Hockey Violent or Non Violent"  # Change this to the actual path
output_path = "D:/Computer Vision/hockey__frames"

# RAFT model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "path_to_pretrained_raft_model/raft-sintel.pth"  # Change to your RAFT model path

# Load RAFT model
model = torch.nn.DataParallel(RAFT())
model.load_state_dict(torch.load(model_path))
model = model.module
model.to(device)
model.eval()

# Create directories for output if they don't already exist
os.makedirs(output_path, exist_ok=True)
temporal_output = os.path.join(output_path, "temporal_frames")
optical_output = os.path.join(output_path, "optical_flow_frames")
os.makedirs(temporal_output, exist_ok=True)
os.makedirs(optical_output, exist_ok=True)

# RAFT helper function
def compute_optical_flow_raft(prev_frame, curr_frame):
    """Computes optical flow using RAFT."""
    prev_frame_torch = torch.from_numpy(prev_frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    curr_frame_torch = torch.from_numpy(curr_frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    
    padder = InputPadder(prev_frame_torch.shape)
    prev_frame_torch, curr_frame_torch = padder.pad(prev_frame_torch, curr_frame_torch)
    
    with torch.no_grad():
        flow = model(prev_frame_torch, curr_frame_torch, iters=20, test_mode=True)[0]
    flow = flow[0].permute(1, 2, 0).cpu().numpy()  # Convert back to NumPy
    return flow

# Iterate through the violent and non-violent directories
for category in ["violent", "non_violent"]:
    category_path = os.path.join(dataset_path, category)
    category_temporal = os.path.join(temporal_output, category)
    category_optical = os.path.join(optical_output, category)

    os.makedirs(category_temporal, exist_ok=True)
    os.makedirs(category_optical, exist_ok=True)
    
    for video_file in tqdm(os.listdir(category_path), desc=f"Processing {category} videos"):
        video_path = os.path.join(category_path, video_file)
        video_name = os.path.splitext(video_file)[0]
        
        # Create folders for each video
        temporal_folder = os.path.join(category_temporal, video_name)
        optical_folder = os.path.join(category_optical, video_name)
        os.makedirs(temporal_folder, exist_ok=True)
        os.makedirs(optical_folder, exist_ok=True)
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        success, prev_frame = cap.read()
        
        if not success:
            print(f"Failed to read {video_file}")
            continue
        
        frame_count = 0
        
        while success:
            # Save the temporal (RGB) frame
            cv2.imwrite(os.path.join(temporal_folder, f"frame_{frame_count}.jpg"), prev_frame)
            
            # Read next frame
            success, curr_frame = cap.read()
            if not success:
                break
            
            # Compute optical flow using RAFT
            flow = compute_optical_flow_raft(prev_frame, curr_frame)
            
            # Visualize optical flow
            hsv = np.zeros_like(prev_frame)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2  # Hue represents direction
            hsv[..., 1] = 255  # Saturation
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Magnitude represents speed
            optical_flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Save the optical flow image
            cv2.imwrite(os.path.join(optical_folder, f"flow_{frame_count}.jpg"), optical_flow_img)
            
            # Update previous frame for the next iteration
            prev_frame = curr_frame
            frame_count += 1
        
        cap.release()

print("Frame extraction complete!")
