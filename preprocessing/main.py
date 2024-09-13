import cv2 as cv
import boto3
import io
import os
from s3_function import load_video_local_storage,distroy_video_from_local_storage,save_single_frame_in_s3,delete_dir_folder
from frameExtraction import frame_extract

s3 = boto3.resource(
    service_name = 's3',
    region_name = 'ap-south-1',
    aws_access_key_id = 'AKIASDRAMZYPZYTJMV6B',
    aws_secret_access_key = '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'
)
bucket_name = 'qbitfacedetection'

ori_video = "original"

for i in range (7,37):
    load_video_local_storage(s3,bucket_name,s3_video_path=f"dataset/{ori_video}_sequences/{ori_video}_video ({i}).mp4")

    frame_extract(s3,bucket_name,ori_video,f"{ori_video}_micro_expresion{i}")
    distroy_video_from_local_storage()

ori_video = "manipulated"

for i in range (1,57):
    load_video_local_storage(s3,bucket_name,s3_video_path=f"dataset/{ori_video}_sequences/{ori_video}_video ({i}).mp4")

    frame_extract(s3,bucket_name,ori_video,f"{ori_video}_micro_expresion{i}")
    distroy_video_from_local_storage()