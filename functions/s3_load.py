import boto3
import numpy as np
import cv2
from tqdm import tqdm

# Initialize S3 client
s3 = boto3.resource(
    service_name = 's3',
    region_name = 'ap-south-1',
    aws_access_key_id = 'AKIASDRAMZYPZYTJMV6B',
    aws_secret_access_key = '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'
)

def load_frames_from_s3(bucket_name, folder_path):
    video_dict = {}

    # List all frame file paths in the folder
    for obj in s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)['Contents']:
        frame_file = obj['Key'].split('/')[-1]
        if not frame_file:  # Skip directories
            continue
        video_name = frame_file.split('_frame')[0]  # Group frames by video name
        if video_name not in video_dict:
            video_dict[video_name] = []
        video_dict[video_name].append(obj['Key'])

    return video_dict

def load_video_frames(bucket_name, video_frames, frame_size):
    frames = []
    for frame_key in video_frames:
        response = s3.get_object(Bucket=bucket_name, Key=frame_key)
        frame = np.frombuffer(response['Body'].read(), np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, frame_size)
        frame = frame.astype('float32') / 255.0
        frames.append(frame)
    return np.array(frames)

def create_dataset(bucket_name, facial_folder, micro_exp_folder, frame_size):
    facial_video_dict = load_frames_from_s3(bucket_name, facial_folder)
    micro_exp_video_dict = load_frames_from_s3(bucket_name, micro_exp_folder)
    
    common_videos = set(facial_video_dict.keys()).intersection(micro_exp_video_dict.keys())

    dataset = []

    for video in tqdm(common_videos, desc="Processing videos"):
        facial_frames = load_video_frames(bucket_name, facial_video_dict[video], frame_size)
        micro_exp_frames = load_video_frames(bucket_name, micro_exp_video_dict[video], frame_size)

        # Label: 0 for original, 1 for manipulated (based on folder path)
        label = 0 if 'original' in facial_folder else 1
        
        dataset.append({
            'video_index': video,
            'facial_frames': facial_frames,
            'micro_expression_frames': micro_exp_frames,
            'label': label
        })

    return dataset

# Parameters
bucket_name = 'qbitfacedetection'
facial_folder = 'frames/original'  # Change to 'manipulated' for manipulated videos
micro_exp_folder = 'Micro_Expressions/original'  # Same here
frame_size = (224, 224)  # Resize frame size

# Create dataset
dataset = create_dataset(bucket_name, facial_folder, micro_exp_folder, frame_size)

# Example of dataset structure
for data in dataset[:3]:  # Show first 3 samples
    print(f"Video: {data['video_index']}, Label: {data['label']}")
    print(f"Facial Frames Shape: {data['facial_frames'].shape}, Micro-expression Frames Shape: {data['micro_expression_frames'].shape}")
