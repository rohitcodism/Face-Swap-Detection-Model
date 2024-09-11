import cv2 as cv
import boto3
import io
import os
from s3_function import load_video_local_storage,distroy_video_from_local_storage,save_single_frame_in_s3,get_frame_folder_from_s3,delete_dir_folder

s3 = boto3.resource(
    service_name = 's3',
    region_name = 'ap-south-1',
    aws_access_key_id = 'AKIASDRAMZYPZYTJMV6B',
    aws_secret_access_key = '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'
)
bucket_name = 'qbitfacedetection'


# load_video_local_storage(s3,bucket_name,s3_video_path=f"dataset/manipulated_sequences/manipulated_video ({1}).mp4")

get_frame_folder_from_s3(s3,bucket_name,"manipulated",1)
delete_dir_folder()

# distroy_video_from_local_storage()