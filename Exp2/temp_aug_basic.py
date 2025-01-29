import os
import cv2
import numpy as np
import random
from tqdm import tqdm

# Basic Augmentation Functions

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

def apply_basic_augmentations(video_file, input_folder, output_folder):
    video_path = os.path.join(input_folder, video_file)
    frames = read_video(video_path)
    fps = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS))
    frame_size = (frames[0].shape[1], frames[0].shape[0])  # Width, Height

    augmentations = {
        "reversed": reverse_frames(frames),
        "speed_changed": change_speed_randomly(frames),
        "cropped": crop_random_clip(frames, clip_length=min(100, len(frames))),
        "frame_dropout": apply_frame_dropout(frames, dropout_rate=0.2),
        "flickered": add_flicker_effect(frames, flicker_rate=0.05),
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

    for video_file in tqdm(video_files, desc="Applying Basic Augmentations", mininterval=5.0):
        apply_basic_augmentations(video_file, input_folder, output_folder)
