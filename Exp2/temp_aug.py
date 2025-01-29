import os
import cv2
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Temporal Augmentation Functions

def reverse_frames(frames):
    return frames[::-1]

def change_speed_randomly(frames, speed_range=(0.5, 2.0)):
    speed_factor = random.uniform(*speed_range)
    indices = np.arange(0, len(frames), speed_factor).astype(int)
    return [frames[i] for i in indices if i < len(frames)]

def crop_random_clip(frames, clip_length):
    if len(frames) <= clip_length:
        return frames
    start_index = random.randint(0, len(frames) - clip_length)
    return frames[start_index:start_index + clip_length]

def apply_frame_dropout(frames, dropout_rate=0.1):
    total_frames = len(frames)
    drop_indices = random.sample(range(total_frames), int(total_frames * dropout_rate))
    return [frame for i, frame in enumerate(frames) if i not in drop_indices]

def add_flicker_effect(frames, flicker_rate=0.05):
    for i in range(len(frames)):
        if random.random() < flicker_rate:
            frames[i] = np.zeros_like(frames[i])  # Flicker by setting frames to black
    return frames

def adaptive_frame_sampling(frames, motion_threshold=0.2):
    sampled_frames = []
    prev_frame = None
    for i, frame in enumerate(frames):
        if prev_frame is not None:
            motion = np.sum(np.abs(frame - prev_frame)) / frame.size
            if motion > motion_threshold:
                sampled_frames.append(frame)
        prev_frame = frame
    return sampled_frames

def apply_frame_masking(frames, mask_ratio=0.2):
    """
    Apply random masks over frames to simulate occlusions.
    mask_ratio determines the percentage of the frame covered by the mask.
    """
    masked_frames = []
    for frame in frames:
        mask_height = int(frame.shape[0] * mask_ratio)
        mask_width = int(frame.shape[1] * mask_ratio)
        x_start = random.randint(0, frame.shape[1] - mask_width)
        y_start = random.randint(0, frame.shape[0] - mask_height)
        
        frame_copy = frame.copy()
        frame_copy[y_start:y_start + mask_height, x_start:x_start + mask_width] = 0  # Black mask
        masked_frames.append(frame_copy)
    return masked_frames

# Utility functions

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def write_video(output_path, frames, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    for frame in frames:
        out.write(frame)
    out.release()

# Augmentation processing

def process_video(video_file, input_folder, output_folder):
    video_path = os.path.join(input_folder, video_file)
    frames = read_video(video_path)
    fps = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS))
    frame_size = (frames[0].shape[1], frames[0].shape[0])  # Width, Height

    # Apply temporal augmentations
    augmentations = {
        "reversed": reverse_frames(frames),
        "speed_changed": change_speed_randomly(frames),
        "cropped": crop_random_clip(frames, clip_length=min(100, len(frames))),  # Crop max 100 frames
        "frame_dropout": apply_frame_dropout(frames, dropout_rate=0.2),
        "flickered": add_flicker_effect(frames, flicker_rate=0.05),
        "masked": apply_frame_masking(frames, mask_ratio=0.2),
        "adaptive_sampled": adaptive_frame_sampling(frames),
    }

    for aug_name, aug_frames in augmentations.items():
        output_filename = f"{os.path.splitext(video_file)[0]}_{aug_name}.mp4"
        output_path = os.path.join(output_folder, output_filename)
        write_video(output_path, aug_frames, fps, frame_size)

# Apply augmentations in batches
def apply_temporal_augmentations(input_folder, output_folder, percentage=50, max_workers=4):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    num_videos_to_augment = int(len(video_files) * (percentage / 100))
    videos_to_augment = random.sample(video_files, num_videos_to_augment)  # Randomly select videos

    print(f"Selected {num_videos_to_augment} videos ({percentage}%) for augmentation out of {len(video_files)} total videos.")

    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(lambda video_file: process_video(video_file, input_folder, output_folder),
                               videos_to_augment), total=len(videos_to_augment), desc="Processing Videos", mininterval=5.0))

# Example Usage
if __name__ == "__main__":
    input_folder = "F:/video_data/deepfakes"  # Replace with your input folder path
    output_folder = "F:/augmented_type_7_videos/deepfakes2"  # Replace with your output folder path
    percentage = 100  # Replace with the percentage of videos to augment
    apply_temporal_augmentations(input_folder, output_folder, percentage, max_workers=4)  # Adjust max_workers as needed
