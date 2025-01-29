import os
import cv2
import numpy as np
import random
from tqdm import tqdm

# Advanced Augmentation Functions

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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    for frame in frames:
        out.write(frame)
    out.release()

def apply_advanced_augmentations(video_file, input_folder, output_folder):
    video_path = os.path.join(input_folder, video_file)
    frames = read_video(video_path)
    fps = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS))
    frame_size = (frames[0].shape[1], frames[0].shape[0])  # Width, Height

    augmentations = {
        "adaptive_sampled": adaptive_frame_sampling(frames),
        "masked": apply_frame_masking(frames, mask_ratio=0.2),
    }

    for aug_name, aug_frames in augmentations.items():
        output_filename = f"{os.path.splitext(video_file)[0]}_{aug_name}.mp4"
        output_path = os.path.join(output_folder, output_filename)
        write_video(output_path, aug_frames, fps, frame_size)

if __name__ == "__main__":
    input_folder = "D:/Projects/Face-Swap-Detection-Model/Faceforensic/manipulated_sequences/Deepfakes/c40/videos"  # Replace with your input folder
    output_folder = "D:/augmented_type_7_videos/df_lq"  # Replace with your output folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

    for video_file in tqdm(video_files, desc="Applying Advanced Augmentations", mininterval=5.0):
        apply_advanced_augmentations(video_file, input_folder, output_folder)
