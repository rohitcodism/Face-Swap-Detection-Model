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


