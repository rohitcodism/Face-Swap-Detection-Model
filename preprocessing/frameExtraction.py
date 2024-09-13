import cv2
import dlib
import os
import random
import numpy as np

from s3_function import save_single_frame_in_s3,save_micro_for_single_frame_in_s3

output_dir = "main_directory"       # Directory setup for parts of face
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

another_dir = "face_directory"      # Derectory setup for whole face
if not os.path.exists(another_dir):
    os.makedirs(another_dir)

video_folder = "preprocessing/videos/"


detector = dlib.get_frontal_face_detector()     # Load Dlib face detector and shape predictor
    
predictor = dlib.shape_predictor("preprocessing/shape_predictor_68_face_landmarks.dat")   # Pre-trained model

desired_size = (64, 64)     # Desired size for resizing 

# Maintaining Aspect Ratio for better quality.......................

def resize_with_aspect_ratio(image, width=None, height=None, inter=None):
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# Augmentation function...........................

def augment_image(image):
    if random.choice([True, False]):    # Random horizontal flip
        image = cv2.flip(image, 1)
        
    angle = random.choice([-10, 0, 10]) # Random rotation (-10, 0, 10 degrees)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h))

    brightness_factor = random.uniform(0.5, 1.5)    # Random brightness adjustment
    image = cv2.convertScaleAbs(image, alpha=1, beta=brightness_factor * 255 - 255)
    
    contrast_factor = random.uniform(0.5, 1.5)      # Random contrast adjustment
    image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)

    return image

# Function to crop in high resolution, resize, and save only the resized version
    
def crop_resize_and_save(image, landmarks, start_idx, end_idx, feature_name, frame_count, i,s3,bucket_name,video_type,folder_name,frame_obj_num,extend):
    
    # Get the landmarks for the feature (e.g., eyes, nose, mouth)
    points = landmarks[start_idx:end_idx]
    
    # Calculate the bounding box for the feature
    x, y, w, h = cv2.boundingRect(np.array(points))
    
    # Check if the bounding box is valid, otherwise adjust
    if w <= 0 or h <= 0:
        print(f"Adjusting {feature_name} bounding box for frame {frame_count} face {i}: Invalid w={w}, h={h}")
        
        # Set a default width and height if bounding box is too small
        w = max(w, 30)  # Use a reasonable minimum width
        h = max(h, 30)  # Use a reasonable minimum height

    # Ensure the bounding box stays within image bounds
    if x < 0: x = 0
    if y < 0: y = 0
    if x + w > image.shape[1]: w = image.shape[1] - x
    if y + h > image.shape[0]: h = image.shape[0] - y

    # Crop the feature from the image (ensure it's within bounds)
    cropped_feature = image[y:y + h, x:x + w]

    # If the crop is empty after adjusting, we can use the whole face as a fallback
    if cropped_feature.size == 0:
        print(f"Empty crop for {feature_name}, using whole face for frame {frame_count} face {i}")
        cropped_feature = image  # Fallback to the whole image or face region

    # Resize the cropped feature to the desired size
    resized_feature = resize_with_aspect_ratio(cropped_feature, 128, 128, inter=cv2.INTER_CUBIC)
    
    # Augment the resized image 
    augmented_resized_feature = augment_image(resized_feature)
    
    # Construct the filename for saving the resized image
    # resized_name = os.path.join(video_output_dir, f"frame{frame_count}face{i}{feature_name}_resized.jpg")
    
    # Save the resized feature image with high quality
    # cv2.imwrite(resized_name, augmented_resized_feature, [cv2.IMWRITE_JPEG_QUALITY, 100])  # Save with high quality
    # print(f"Saved {resized_name}")
    save_micro_for_single_frame_in_s3(s3,bucket_name,resized_feature,video_type,folder_name,frame_obj_num,extend)


def crop_and_save_full_face(s3,bucket_name,image, face, frame_count, i, desired_size,video_type,folder_name):
    # Extract face coordinates
    x, y, w, h = face.left(), face.top(), face.width(), face.height()

    # Ensure coordinates are within the image boundaries
    if x < 0: x = 0
    if y < 0: y = 0
    if x + w > image.shape[1]: w = image.shape[1] - x
    if y + h > image.shape[0]: h = image.shape[0] - y

    # Crop the face from the frame
    cropped_face = image[y:y + h, x:x + w]

    # Augment the resized image 
    augmented_resized_feature = augment_image(cropped_face)

    # Normalize the pixel values to the range [0, 1]
    normalized_face = augmented_resized_feature.astype("float32") / 255.0

    # Display the normalized face (optional)
    # cv2.imshow(f"Normalized Face {i}", normalized_face)

    # Resize the cropped face (check if not empty before resizing)
    if cropped_face.size != 0:
        resized_face = cv2.resize(augmented_resized_feature, desired_size, interpolation=cv2.INTER_LANCZOS4)
        save_single_frame_in_s3(s3,bucket_name,resized_face,video_type,folder_name,frame_count)
        # Save the resized face to the s3
        
# Main loop to process videos
def frame_extract(s3,bucket_name,video_type,folder_name):
    for video_file in os.listdir(video_folder):
        
        if video_file.endswith(('.mp4', '.avi', '.mov')):  # Process video files with these extensions
            video_path = os.path.join(video_folder, video_file)
            print(f"Processing video: {video_path}")
            
            video_name = os.path.splitext(video_file)[0]
            video_output_dir = os.path.join(output_dir, video_name)
            video_output_dir1 = os.path.join(another_dir, video_name)
            
            if not os.path.exists(video_output_dir):
                os.makedirs(video_output_dir)
            if not os.path.exists(video_output_dir1):
                os.makedirs(video_output_dir1)
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # total frames in one video
            print(total_frames);
            if (total_frames<=700):
                desired_frame_count = max(30, (total_frames * 20) // 100)   # required frame
                n = max(1, total_frames // desired_frame_count)
            elif(700<total_frames<=1400):
                desired_frame_count = max(30, (total_frames * 15) // 100)   # required frame
                n = max(1, total_frames // desired_frame_count)
            else:
                desired_frame_count = max(30, (total_frames * 10) // 100)   # required frame
                n = max(1, total_frames // desired_frame_count)
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % n != 0:
                    continue
                
                faces = detector(frame)
                
                for i, face in enumerate(faces):
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    new_frame_count=10000+(frame_count//n)
                    # Draw a rectangle around the detected face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    crop_and_save_full_face(s3,bucket_name,frame,face,new_frame_count,{i},(224,224),video_type,folder_name)

                    # Detect landmarks
                    shape = predictor(frame, face)
                    landmarks = [(shape.part(j).x, shape.part(j).y) for j in range(68)]
                    
                    
                    # Crop, resize, and save left eye, right eye, nose, and mouth
                    crop_resize_and_save(frame, landmarks, 18, 48, "left_eye", frame_count, i,s3,bucket_name,video_type,folder_name,new_frame_count,"eye")
                    #crop_resize_and_save(frame, landmarks, 42, 48, "right_eye", video_output_dir, frame_count, i)
                    crop_resize_and_save(frame, landmarks, 27, 36, "nose", frame_count,i,s3,bucket_name,video_type,folder_name,new_frame_count,"nose")
                    crop_resize_and_save(frame, landmarks, 48, 68, "mouth", frame_count, i,s3,bucket_name,video_type,folder_name,new_frame_count,"mouth")
                        
                # # Display the frame with the face detection
                # cv2.imshow("Detected Faces", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            cap.release()
    cv2.destroyAllWindows()