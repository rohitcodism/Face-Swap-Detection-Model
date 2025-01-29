import os

def count_folders(directory):
    # List all items in the directory
    items = os.listdir(directory)
    
    # Count the number of directories
    folder_count = sum(1 for item in items if os.path.isdir(os.path.join(directory, item)))
    
    return folder_count

# Example usage
directory_path = "G:/Preprocessed_data/original/facial_frames"  # Replace with the path to your directory
folder_count = count_folders(directory_path)
print(f"Number of folders in '{directory_path}': {folder_count}")
