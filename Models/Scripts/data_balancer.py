import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def copy_folder(src, dest):
    """Function to copy a folder from source to destination."""
    shutil.copytree(src, dest)
    print(f"Copied {os.path.basename(src)} to {dest}")

def copy_balanced_data(src_base, dest_base, num_samples=2190, max_workers=8):
    categories = ['manipulated', 'original']
    frame_types = ['facial_frames', 'micro_expression_frames']

    tasks = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for category in categories:
            src_category_path = os.path.join(src_base, category)
            dest_category_path = os.path.join(dest_base, category)

            os.makedirs(dest_category_path, exist_ok=True)

            for frame_type in frame_types:
                src_frame_path = os.path.join(src_category_path, frame_type)
                dest_frame_path = os.path.join(dest_category_path, frame_type)

                os.makedirs(dest_frame_path, exist_ok=True)

                # Get list of video folders and sort them alphabetically
                video_folders = sorted(
                    f for f in os.listdir(src_frame_path)
                    if os.path.isdir(os.path.join(src_frame_path, f))
                )

                # For the manipulated category, take the first `num_samples` videos
                if category == 'manipulated':
                    video_folders = video_folders[:num_samples]

                # Submit each copy task to the thread pool
                for video_folder in video_folders:
                    src_video_path = os.path.join(src_frame_path, video_folder)
                    dest_video_path = os.path.join(dest_frame_path, video_folder)
                    
                    tasks.append(executor.submit(copy_folder, src_video_path, dest_video_path))

        # Progress tracking with tqdm
        with tqdm(total=len(tasks), desc="Copying folders", unit="folder") as progress_bar:
            for future in as_completed(tasks):
                future.result()  # Wait for each task to complete
                progress_bar.update(1)  # Update progress bar after each task

# Define source and destination paths
src_base_path = 'G:/Preprocessed_data'  # Replace with actual path
dest_base_path = 'G:/Final_data'  # Replace with actual path

copy_balanced_data(src_base_path, dest_base_path)
