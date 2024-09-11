import cv2 as cv
import boto3
import shutil
from PIL import Image
import io
import os
import numpy as np

video_local = 'videos/samp.mp4'

def load_video_local_storage(s3,bucket_name,s3_video_path):
    video_obj = io.BytesIO()
    s3.Bucket(bucket_name).download_fileobj(s3_video_path, video_obj)
    print('OK')
    with open(video_local, 'wb') as f:
        f.write(video_obj.getbuffer())

def distroy_video_from_local_storage():
    os.remove(video_local)
    
def save_single_frame_in_s3(s3,bucket_name,image_obj,video_type,folder_name,frame_obj_num):
    success, encoded_image = cv.imencode('.jpg', image_obj)
        
    if not success:
        print(f"Failed to encode frame {image_obj}")
        return
    image_obj = io.BytesIO(encoded_image.tobytes())
    image_key = f'frames/{video_type}/{folder_name}/frame_{frame_obj_num}.jpg'
    s3.Bucket(bucket_name).upload_fileobj(image_obj, image_key)
    
def save_all_frame_of_a_video_in_s3(s3,bucket_name,video_path):
    # code for preprocessing
    return

def get_frame_folder_from_s3(s3,bucket_name,s3_folder_type,s3_folder_num):
    s3_folder=f"frames/{s3_folder_type}/{s3_folder_type}_frames{s3_folder_num}"
    local_dir = 'preprocessing/dir/'

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    # Select the S3 bucket
    bucket = s3.Bucket(bucket_name)

    # Iterate over all objects in the S3 folder
    for obj in bucket.objects.filter(Prefix=s3_folder):
        # Get the file key (path in S3)
        s3_key = obj.key

        # Skip folders, only process files
        if not s3_key.endswith('/'):
            # Define the local file path
            relative_path = os.path.relpath(s3_key, s3_folder)
            local_file_path = os.path.join(local_dir, relative_path)

            # Ensure the directory structure is maintained locally
            local_file_dir = os.path.dirname(local_file_path)
            if not os.path.exists(local_file_dir):
                os.makedirs(local_file_dir)

            # Download the file into memory
            s3_object = s3.Object(bucket_name, s3_key)
            file_stream = io.BytesIO()
            s3_object.download_fileobj(file_stream)

            # Reset file pointer to start
            file_stream.seek(0)

            # Open the image using PIL
            img = Image.open(file_stream)

            # Convert to OpenCV format if needed
            cv_image = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)

            # Save the image locally
            cv.imwrite(local_file_path, cv_image)
            print(f"Downloaded and saved {s3_key} to {local_file_path}")
            
def delete_dir_folder():
    try:
        shutil.rmtree('preprocessing/dir')
        print(f"Folder {'preprocessing/dir'} and its contents deleted successfully.")
    except OSError as e:
        print(f"Error: {e}")