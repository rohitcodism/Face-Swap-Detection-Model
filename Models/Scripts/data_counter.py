import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def count_subfolders_and_files(path):
    """Function to count subfolders and files in a directory."""
    total_subfolders = 0
    total_files = 0

    # Use os.walk to iterate over directories and files
    for root, dirs, files in os.walk(path):
        total_subfolders += len(dirs)
        total_files += len(files)

    return total_subfolders, total_files

def parallel_folder_counter(base_path, max_workers=16):
    """Function to count files and subfolders in parallel using multithreading."""
    total_folders = 0
    total_files = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder)
            if os.path.isdir(folder_path):
                # Submit each folder to be counted in parallel
                futures.append(executor.submit(count_subfolders_and_files, folder_path))

        for future in as_completed(futures):
            subfolders, files = future.result()
            total_folders += subfolders
            total_files += files

    print(f"Total Subfolders: {total_folders}")
    print(f"Total Files: {total_files}")

# Run the function with the base path of your choice
base_path = "G:/Preprocessed_data"
parallel_folder_counter(base_path)
