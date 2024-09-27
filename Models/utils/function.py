import boto3
from io import BytesIO
from PIL import Image
import numpy as np
import os
import pickle

def load_and_save_data_incrementally(base_path, batch_size=100, save_path='F:/video_data_large_2.pkl'):
    main_folders = ['original_videos', 'manipulated_videos']
    data_types = ['frames', 'micro_expressions']

    # Initialize video data structure
    video_data = {}
    x = 1

    for main_folder in main_folders:
        label = 0 if main_folder == 'original_videos' else 1
        folder_path = os.path.join(base_path, main_folder)

        for data_type in data_types:
            data_type_path = os.path.join(folder_path, data_type)

            for video_folder in os.listdir(data_type_path):
                video_folder_path = os.path.join(data_type_path, video_folder)

                if os.path.isdir(video_folder_path):
                    video_name = video_folder

                    # Initialize if video_name does not exist
                    if video_name not in video_data:
                        video_data[video_name] = {
                            'frames': [],
                            'frame_label': [],
                            'Micro_Expression': [],
                            'Micro_Expression_label': []
                        }

                    frame_files = os.listdir(video_folder_path)

                    # Process files in batches
                    for i in range(0, len(frame_files), batch_size):
                        batch_files = frame_files[i:i + batch_size]

                        for frame_file in batch_files:
                            frame_path = os.path.join(video_folder_path, frame_file)

                            try:
                                with Image.open(frame_path) as img:
                                    if data_type == 'frames':
                                        video_data[video_name]['frames'].append(img.copy())
                                        video_data[video_name]['frame_label'].append(label)
                                    elif data_type == 'micro_expressions':
                                        video_data[video_name]['Micro_Expression'].append(img.copy())
                                        video_data[video_name]['Micro_Expression_label'].append(label)

                                    print(f"{video_name}  {x}")
                                    x += 1

                            except Exception as e:
                                print(f"Error loading image {frame_path}: {e}")

                        # Save the current state to disk
                        with open(save_path, 'ab') as f:
                            pickle.dump({video_name: video_data[video_name]}, f)

                        # Clear the frames to free memory
                        video_data[video_name]['frames'].clear()
                        video_data[video_name]['Micro_Expression'].clear()

    print("Data loading and saving completed.")


def load_4d_array_from_hdd(base_path, batch_size=100):
    main_folders = ['original_videos', 'manipulated_videos']
    data_types = ['frames', 'micro_expressions']

    # Initialize video data structure (if needed to collect some specific data)
    video_data = {}
    x = 1

    for main_folder in main_folders:
        label = 0 if main_folder == 'original_videos' else 1
        folder_path = os.path.join(base_path, main_folder)

        for data_type in data_types:
            data_type_path = os.path.join(folder_path, data_type)

            for video_folder in os.listdir(data_type_path):
                video_folder_path = os.path.join(data_type_path, video_folder)

                if os.path.isdir(video_folder_path):
                    video_name = video_folder

                    if video_name not in video_data:
                        video_data[video_name] = {
                            'frames': [],
                            'frame_label': [],
                            'Micro_Expression': [],
                            'Micro_Expression_label': []
                        }

                    frame_files = os.listdir(video_folder_path)

                    # Process files in batches to avoid memory overload
                    for i in range(0, len(frame_files), batch_size):
                        batch_files = frame_files[i:i + batch_size]

                        for frame_file in batch_files:
                            frame_path = os.path.join(video_folder_path, frame_file)

                            # Use a context manager to open and process the image, then immediately close it
                            with Image.open(frame_path) as img:
                                if data_type == 'frames':
                                    # Store only essential data
                                    video_data[video_name]['frames'].append(img.copy())
                                    video_data[video_name]['frame_label'].append(label)
                                elif data_type == 'micro_expressions':
                                    video_data[video_name]['Micro_Expression'].append(img.copy())
                                    video_data[video_name]['Micro_Expression_label'].append(label)

                            print(f"{video_name}  {x}")
                            x += 1

                        # # Clear memory after each batch to avoid running out of memory
                        # video_data[video_name]['frames'].clear()
                        # video_data[video_name]['Micro_Expression'].clear()

    return video_data



def load_4d_array_from_s3(bucket_name, region_name, aws_access_key_id, aws_secret_access_key):
    # Initialize boto3 S3 resource
    s3 = boto3.resource(
        service_name='s3',
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    # Get the S3 bucket object
    bucket = s3.Bucket(bucket_name)

    # The base directories to search in
    base_folders = ['frames', 'Micro_Expression']
    sub_folders = ['original', 'manipulated']

    # Initialize the 4D array
    video_data = {}
    x = 1
    # Loop through the base folders (frames and micro)
    for base_folder in base_folders:
        for sub_folder in sub_folders:
            # Filter the objects based on the current folder structure
            prefix = f'{base_folder}/{sub_folder}/'
            for obj in bucket.objects.filter(Prefix=prefix):
                key = obj.key

                # Skip if it's a folder
                if key.endswith('/'):
                    continue

                # Extract the video name (the part after 'original/' or 'manipulated/')
                video_name = key.split('/')[2]

                # Read the frame image
                file_obj = obj.get()
                file_content = file_obj['Body'].read()
                img = Image.open(BytesIO(file_content))

                # Create the video name dictionary if it doesn't exist
                if video_name not in video_data:
                    video_data[video_name] = {'frames': [], 'frame_label': [], 'Micro_Expression': [],
                                            'Micro_Expression_label': []}

                # Determine if it's from the 'original' or 'manipulated' folder
                label = 0 if sub_folder == 'original' else 1

                # Add the frame to the corresponding array
                if base_folder == 'frames':
                    video_data[video_name]['frames'].append(img)
                    video_data[video_name]['frame_label'].append(label)
                elif base_folder == 'Micro_Expression':
                    video_data[video_name]['Micro_Expression'].append(img)
                    video_data[video_name]['Micro_Expression_label'].append(label)
                print(f"{video_name}  {x}")
                x = x + 1

    return video_data


def load_dataset():
    bucket_name = 'qbitfacedetection'
    region_name = 'ap-south-1'
    aws_access_key_id = 'AKIASDRAMZYPZYTJMV6B'
    aws_secret_access_key = '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'

    # Call the function to load data
    video_data = load_4d_array_from_s3(bucket_name, region_name, aws_access_key_id, aws_secret_access_key)

    # Now video_data contains all the images organized in a 4D array
    return video_data
