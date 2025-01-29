import os
import shutil
import math
from tqdm import tqdm

# Define source and destination folders
src_folder = "G:/Big_data/manipulated_videos"
dest_folders = [
    "G:/Datum/Piyush/manipulated",
    "G:/Datum/Rohit/manipulated",
    "G:/Datum/Koley/manipulated",
    "G:/Datum/Dipayan/manipulated"
]

# Get list of all video files in the source folder
all_videos = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
total_videos = len(all_videos)

# Calculate number of videos per folder
videos_per_folder = math.ceil(total_videos / len(dest_folders))

# Create destination folders if they don't exist
for folder in dest_folders:
    os.makedirs(folder, exist_ok=True)

# Distribute videos equally into destination folders with progress bar
with tqdm(total=total_videos, desc="Distributing Videos") as pbar:
    for i, video in enumerate(all_videos):
        dest_folder = dest_folders[i % len(dest_folders)]
        src_video_path = os.path.join(src_folder, video)
        dest_video_path = os.path.join(dest_folder, video)
        shutil.move(src_video_path, dest_video_path)
        pbar.update(1)

print(f"Distributed {total_videos} videos equally among the 4 folders.")
