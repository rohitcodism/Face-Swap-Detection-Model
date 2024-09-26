import boto3
from io import BytesIO
from PIL import Image
import numpy as np
import os


def load_4d_array_from_hdd(base_path):
    # Define the main categories (original and manipulated)
    main_folders = ['original_videos', 'manipulated_videos']
    data_types = ['frames', 'micro_expressions']

    # Initialize the 4D array
    video_data = {}
    x = 1

    # Loop through the main folders (original and manipulated)
    for main_folder in main_folders:
        # Determine the label (0 for original, 1 for manipulated)
        label = 0 if main_folder == 'original_videos' else 1

        # Construct the path for the current main folder
        folder_path = os.path.join(base_path, main_folder)

        # Iterate through frames and micro_expressions folders
        for data_type in data_types:
            data_type_path = os.path.join(folder_path, data_type)

            # Loop through each video folder within frames or micro_expressions
            for video_folder in os.listdir(data_type_path):
                video_folder_path = os.path.join(data_type_path, video_folder)

                # Ensure it's a directory
                if os.path.isdir(video_folder_path):
                    video_name = video_folder  # Use the folder name as the video name

                    # Initialize the video dictionary if it doesn't exist
                    if video_name not in video_data:
                        video_data[video_name] = {
                            'frames': [],
                            'frame_label': [],
                            'Micro_Expression': [],
                            'Micro_Expression_label': []
                        }

                    # Iterate through the frames or micro-expression images
                    for frame_file in os.listdir(video_folder_path):
                        frame_path = os.path.join(video_folder_path, frame_file)

                        # Load the frame image
                        img = Image.open(frame_path)

                        # Add the frame to the corresponding array
                        if data_type == 'frames':
                            video_data[video_name]['frames'].append(img)
                            video_data[video_name]['frame_label'].append(label)
                        elif data_type == 'micro_expressions':
                            video_data[video_name]['Micro_Expression'].append(img)
                            video_data[video_name]['Micro_Expression_label'].append(label)

                        print(f"{video_name}  {x}")
                        x = x + 1

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
