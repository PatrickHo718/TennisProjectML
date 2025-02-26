import os
import cv2
import numpy as np
import torch
from tracknet import BallTrackerNet
import torch.nn.functional as F
from tqdm import tqdm
from postprocess import postprocess, refine_kps
from homography import get_trans_matrix, refer_kps

# Load the model
model = BallTrackerNet(out_channels=15)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.load_state_dict(torch.load('model_tennis_court_det.pt', map_location=device))  # Change to your actual model path
model.eval()

# Video file path
video_path = 'tennisvid2.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get total frame count
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Trackbar callback function
def set_frame(val):
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)  # Set the video position based on slider value

# Create a window
cv2.namedWindow('Ball Tracking')

# Create a trackbar (slider)
cv2.createTrackbar('Frame', 'Ball Tracking', 0, total_frames - 1, set_frame)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Resize the frame to match model input size
    frame_resized = cv2.resize(frame, (640, 360))
    
    # Preprocess frame for model input
    inp = (frame_resized.astype(np.float32) / 255.0)  # Normalize
    inp = torch.tensor(np.rollaxis(inp, 2, 0)).unsqueeze(0).to(device)  # Convert (H, W, C) → (C, H, W) and add batch dim
    
    # Run model inference
    with torch.no_grad():
        out = model(inp.float())[0]  # Get model output
        pred = F.sigmoid(out).cpu().numpy()  # Apply sigmoid activation
    
    # Process predictions
    points = []
    for kps_num in range(14):  # Assuming 14 keypoints
        heatmap = (pred[kps_num] * 255).astype(np.uint8)
        x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
        points.append((x_pred, y_pred))

    # Draw detected points on the frame
    for x, y in points:
        if x is not None and y is not None:
            frame = cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)  # Draw red circles

    # Display the frame
    cv2.imshow('Ball Tracking', frame)

    # Update trackbar position
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos('Frame', 'Ball Tracking', current_frame)

    # Handle keyboard input
    key = cv2.waitKey(25) & 0xFF
    if key == ord('q'):  # Quit on 'q' key
        break
    elif key == ord('p'):  # Pause on 'p' key
        cv2.waitKey(-1)  # Wait indefinitely until a key is pressed

# Release resources
cap.release()
cv2.destroyAllWindows()