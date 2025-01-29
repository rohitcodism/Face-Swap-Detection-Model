from io import BytesIO
import os
from tqdm import tqdm
from PIL import Image
import pickle

def pil_to_bytes(pil_img):
    with BytesIO() as buffer:
        pil_img.save(buffer, format='JPEG')
        return buffer.getvalue()


def load_and_save_data(base_path, save_path='D:/Projects/Face-Swap-Detection-Model/video_data_fs_16K.pkl'):
    main_folders = ['og', 'f2f']
    data_types = ['facial', 'micro_expression']

    # Initialize video data structure
    video_data = {}

    # Add progress bar for main folders
    for main_folder in tqdm(main_folders, desc="Processing Main Folders", mininterval=0.5):  
        label = 0 if main_folder == 'og' else 1
        folder_path = os.path.join(base_path, main_folder)

        # Add progress bar for data types (facial frames, micro expression frames)
        for data_type in tqdm(data_types, desc=f"Processing Data Types ({main_folder})", leave=False, mininterval=1.0):  
            data_type_path = os.path.join(folder_path, data_type)

            # Get video folders and initialize progress bar for them
            video_folders = [f for f in os.listdir(data_type_path) if os.path.isdir(os.path.join(data_type_path, f))]
            for video_folder in tqdm(video_folders, desc=f"{main_folder} - {data_type}", leave=False, mininterval=3.0):
                video_folder_path = os.path.join(data_type_path, video_folder)
                video_name = video_folder

                # Initialize video entry
                if video_name not in video_data:
                    video_data[video_name] = {
                        'frames': [],
                        'frame_label': [],
                        'Micro_Expression': [],
                        'Micro_Expression_label': []
                    }

                frame_files = os.listdir(video_folder_path)

                # Progress bar for frame files
                for frame_file in tqdm(frame_files, desc=f"Processing Frames ({video_name})", leave=False, miniters=10):
                    frame_path = os.path.join(video_folder_path, frame_file)

                    try:
                        # Use a context manager to open and process the image
                        with Image.open(frame_path) as img:
                            if data_type == 'facial':
                                # Convert the image to a byte array (JPEG) before appending
                                video_data[video_name]['frames'].append(pil_to_bytes(img))
                                video_data[video_name]['frame_label'].append(label)
                            elif data_type == 'micro_expression':
                                # Convert the image to a byte array (JPEG) before appending
                                video_data[video_name]['Micro_Expression'].append(pil_to_bytes(img))
                                video_data[video_name]['Micro_Expression_label'].append(label)
                    except Exception as e:
                        # Skip the file and log the issue
                        tqdm.write(f"Skipping file {frame_file} due to error: {e}")
                        continue

    # Save the entire dataset at once to a pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(video_data, f)

    print("Data loading and saving completed.")

# Call the function
load_and_save_data("D:/Projects/Face-Swap-Detection-Model/X_Data")
