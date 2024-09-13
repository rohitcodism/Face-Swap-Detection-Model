import cv2 as cv
import boto3
import shutil
from PIL import Image
import io
import os
import numpy as np

video_local = 'preprocessing/videos/samp.mp4'

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
            
def delete_dir_folder():
    try:
        shutil.rmtree('preprocessing/dir')
        print(f"Folder {'preprocessing/dir'} and its contents deleted successfully.")
    except OSError as e:
        print(f"Error: {e}")
def save_micro_for_single_frame_in_s3(s3,bucket_name,image_obj,video_type,folder_name,frame_obj_num,extend):
    success, encoded_image = cv.imencode('.jpg', image_obj)
        
    if not success:
        print(f"Failed to encode frame {image_obj}")
        return
    image_obj = io.BytesIO(encoded_image.tobytes())
    image_key = f'Micro_Expression/{video_type}/{folder_name}/frame_{frame_obj_num}_{extend}.jpg'
    s3.Bucket(bucket_name).upload_fileobj(image_obj, image_key)