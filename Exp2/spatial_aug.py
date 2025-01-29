import os
import cv2
import numpy as np
import random
from tqdm import tqdm

# Spatial Augmentation Functions
def apply_random_occlusion(frame, block_size=(30, 30)):
    """
    Apply random block occlusion to a frame.
    """
    h, w, _ = frame.shape
    x1 = random.randint(0, w - block_size[0])
    y1 = random.randint(0, h - block_size[1])
    frame[y1:y1 + block_size[1], x1:x1 + block_size[0]] = 0  # Black block
    return frame

def apply_gaussian_noise(frame, mean=0, std=10):
    """
    Apply Gaussian noise to a frame.
    """
    noise = np.random.normal(mean, std, frame.shape).astype(np.uint8)
    noisy_frame = cv2.add(frame, noise)
    return noisy_frame

def adjust_brightness(frame, brightness_factor=1.5):
    """
    Adjust the brightness of a frame.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
    brightened_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return brightened_frame

# Function to apply augmentations to a single frame
def apply_augmentations(frame):
    """
    Apply all selected augmentations to a single frame.
    """
    frame_occluded = apply_random_occlusion(frame)
    frame_noisy = apply_gaussian_noise(frame)
    frame_brightened = adjust_brightness(frame)

    return [frame_occluded, frame_noisy, frame_brightened]

# Process frames in a video folder
def process_video_folder(video_folder):
    """
    Apply augmentations to all frames in a video folder and save them.
    """
    frame_files = [f for f in os.listdir(video_folder) if f.endswith(('.jpg', '.png'))]

    for frame_file in tqdm(frame_files, desc=f"Processing {video_folder}"):
        frame_path = os.path.join(video_folder, frame_file)
        frame = cv2.imread(frame_path)

        # Apply augmentations
        augmented_frames = apply_augmentations(frame)

        # Save augmented frames
        for idx, aug_frame in enumerate(augmented_frames):
            aug_name = ["occluded", "noisy", "brightened"][idx]
            aug_file_name = f"{os.path.splitext(frame_file)[0]}_{aug_name}.jpg"
            aug_file_path = os.path.join(video_folder, aug_file_name)
            cv2.imwrite(aug_file_path, aug_frame)

# Apply augmentations to the entire dataset
def apply_spatial_augmentations_to_dataset(dataset_path):
    """
    Traverse the dataset and apply augmentations to each video's frames.
    """
    for category in ["original", "manipulated"]:  # Assuming 'original' and 'manipulated' folders exist
        category_path = os.path.join(dataset_path, category, "facial")
        if not os.path.exists(category_path):
            print(f"Category path not found: {category_path}")
            continue

        video_folders = [os.path.join(category_path, folder) for folder in os.listdir(category_path)
                         if os.path.isdir(os.path.join(category_path, folder))]

        for video_folder in video_folders:
            process_video_folder(video_folder)

# Example Usage
if __name__ == "__main__":
    dataset_path = "F:/deepfake_dataset"  # Path to your dataset folder
    apply_spatial_augmentations_to_dataset(dataset_path)
