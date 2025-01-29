import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Existing augmentations
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

# New augmentations
def shuffle_frames(frames):
    """
    Randomly shuffles the order of frames in a video.
    """
    shuffled_frames = frames.copy()
    random.shuffle(shuffled_frames)
    return shuffled_frames

def add_flicker_effect(frames, flicker_rate=0.05):
    """
    Introduces flickering by inserting black frames at random intervals.
    """
    flickered_frames = []
    for frame in frames:
        if random.random() < flicker_rate:
            flickered_frames.append(np.zeros_like(frame))  # Add a blank black frame
        flickered_frames.append(frame)
    return flickered_frames

def apply_motion_blur(frames, max_kernel_size=5):
    blurred_frames = []
    for frame in frames:
        kernel_size = random.randint(3, max_kernel_size)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[(kernel_size - 1) // 2, :] = np.ones(kernel_size)
        kernel /= kernel_size
        blurred_frame = cv2.filter2D(frame, -1, kernel)
        blurred_frames.append(blurred_frame)
    return blurred_frames

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

    # Apply augmentations
    augmentations = {
        "reversed": reverse_frames(frames),
        "speed_changed": change_speed_randomly(frames),
        "cropped": crop_random_clip(frames, clip_length=min(100, len(frames))),
        "frame_dropout": apply_frame_dropout(frames, dropout_rate=0.2),
        "shuffled": shuffle_frames(frames),  # Added Frame Shuffling
        "flickered": add_flicker_effect(frames, flicker_rate=0.05),  # Added Flickering Effect
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
    input_folder = "F:/video_data/neuraltextures"  # Replace with your input folder path
    output_folder = "F:/augmented_type_5_videos/neuraltextures"  # Replace with your output folder path
    percentage = 95  # Replace with the percentage of videos to augment
    apply_temporal_augmentations(input_folder, output_folder, percentage, max_workers=4)  # Adjust max_workers as needed
