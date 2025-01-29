import cv2
import dlib
import os
import random
import numpy as np

# ----------------------------- Configuration -----------------------------

# Directory setup for parts of the face
output_dir = "F:/Preprocessed Data/manipulated_preprocessed/micro_expression_frames"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Directory setup for whole face
another_dir = "F:/Preprocessed Data/manipulated_preprocessed/facial_frames"
if not os.path.exists(another_dir):
    os.makedirs(another_dir)

# Directory containing the videos to process
video_folder = r"Models\data\Big_data\manipulated_videos"

# Path to the Dlib shape predictor model
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"  # Ensure this path is correct

# Starting index (0-based). To start from the 588th video, set to 587
start_index = 587

# Desired size for resizing the features
desired_size_features = (64, 64)  # For facial parts
desired_size_full_face = (224, 224)  # For whole face

# Path to the OpenCV window (optional)
display_window = False  # Set to True to display frames during processing

# -------------------------------------------------------------------------

# Function to find all video files in the specified folder and its subfolders
def find_video_files(root_folder):
    video_extensions = ('.mp4', '.avi', '.mov')
    video_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(root, file))
    return video_files

# Initialize Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)  # Load the pre-trained model

# Function to resize image while maintaining aspect ratio
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_LINEAR):
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

# Function to add Gaussian blur
def add_blur(image):
    kernel_size = random.choice([(3, 3), (5, 5), (7, 7)])
    return cv2.GaussianBlur(image, kernel_size, 0)

# Function to add Gaussian noise
def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    sigma = 0.05 ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy_image = image + gauss.reshape(row, col, ch) * 255  # Scale noise
    noisy_image = np.clip(noisy_image, 0, 255).astype('uint8')
    return noisy_image

# Function to convert image to grayscale
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Augmentation function to create augmented versions of an image
def augment_image(image):
    augmented_images = []
    
    # Original
    augmented_images.append(("original", image))
    
    # Random horizontal flip
    image_hor = cv2.flip(image, 1)
    augmented_images.append(("flipped_horizontal", image_hor))
        
    # Random vertical flip
    image_ver = cv2.flip(image, 0)
    augmented_images.append(("flipped_vertical", image_ver))
    
    # Rotation by -45 degrees
    h, w = image.shape[:2]
    M_left = cv2.getRotationMatrix2D((w // 2, h // 2), -45, 1)
    rotate_left = cv2.warpAffine(image, M_left, (w, h))
    augmented_images.append(("rotated_left", rotate_left))
    
    # Rotation by +45 degrees
    M_right = cv2.getRotationMatrix2D((w // 2, h // 2), 45, 1)
    rotated_right = cv2.warpAffine(image, M_right, (w, h))
    augmented_images.append(("rotated_right", rotated_right))
    
    # Random brightness and contrast adjustments
    brightness_factor = random.uniform(0.5, 1.5)
    contrast_factor = random.uniform(0.5, 1.5)
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=brightness_factor * 255 - 255)
    augmented_images.append(("brightness_contrast_adjusted", adjusted_image))
    
    # Apply Gaussian blur
    blurred_image = add_blur(image)
    augmented_images.append(("blurred", blurred_image))
    
    # Apply Gaussian noise
    noisy_image = add_noise(image)
    augmented_images.append(("noisy", noisy_image))
    
    # Convert to grayscale
    grayscale_image = convert_to_grayscale(image)
    augmented_images.append(("grayscale", grayscale_image))
    
    return augmented_images

# Function to crop, resize, pad, augment, and save facial features
def crop_resize_and_save(image, landmarks, start_idx, end_idx, feature_name, video_output_dir, frame_count, face_idx, desired_size):
    
    points = landmarks[start_idx:end_idx]
    
    # Compute the bounding rectangle for the specified landmarks
    x, y, w, h = cv2.boundingRect(np.array(points))

    # Adjust invalid bounding box dimensions
    if w <= 0 or h <= 0:
        print(f"[Warning] Adjusting {feature_name} bounding box for frame {frame_count} face {face_idx}: Invalid w={w}, h={h}")
        w = max(w, 30)
        h = max(h, 30)

    # Ensure bounding box is within image boundaries
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    # Crop the feature from the image
    cropped_feature = image[y:y + h, x:x + w]

    # Handle case where cropping results in an empty image
    if cropped_feature.size == 0:
        print(f"[Warning] Empty crop for {feature_name}, using whole face for frame {frame_count} face {face_idx}")
        cropped_feature = image

    # Maintain aspect ratio while resizing
    h_crop, w_crop = cropped_feature.shape[:2]
    aspect_ratio = w_crop / h_crop

    # Compute new dimensions while maintaining aspect ratio
    if aspect_ratio > 1:  # Width is greater than height
        new_w = desired_size[0]
        new_h = int(new_w / aspect_ratio)
    else:  # Height is greater than width
        new_h = desired_size[1]
        new_w = int(new_h * aspect_ratio)

    # Resize the cropped feature
    resized_feature = cv2.resize(cropped_feature, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Create a blank image of the desired size and center the resized feature on it
    if len(resized_feature.shape) == 2:
        # Grayscale image
        padded_feature = np.full((desired_size[1], desired_size[0]), 0, dtype=np.uint8)
    else:
        # Color image
        padded_feature = np.full((desired_size[1], desired_size[0], 3), 0, dtype=np.uint8)

    # Compute padding to center the image
    x_offset = (desired_size[0] - new_w) // 2
    y_offset = (desired_size[1] - new_h) // 2

    # Paste the resized feature into the padded image
    padded_feature[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_feature

    # Augment the resized and padded image
    augmented_images = augment_image(padded_feature)
    for augment_type, augmented_resized_feature in augmented_images:
        # Construct the filename
        if len(augmented_resized_feature.shape) == 2:
            # If grayscale, save as single channel
            file_extension = '.jpg'
        else:
            file_extension = '.jpg'
        
        resized_name = os.path.join(
            video_output_dir,
            f"frame{frame_count}_face{face_idx}_{feature_name}_{augment_type}{file_extension}"
        )
        
        # Save the augmented image
        if len(augmented_resized_feature.shape) == 2:
            # Grayscale image saved with single channel
            cv2.imwrite(resized_name, augmented_resized_feature, [cv2.IMWRITE_JPEG_QUALITY, 100])
        else:
            # Color image
            cv2.imwrite(resized_name, augmented_resized_feature, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        print(f"Saved {resized_name}")

# Function to crop, resize, augment, and save the full face
def crop_and_save_full_face(image, face, video_output_dir1, frame_count, face_idx, desired_size):
    x, y, w, h = face.left(), face.top(), face.width(), face.height()

    # Ensure bounding box is within image boundaries
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    # Crop the face from the image
    cropped_face = image[y:y + h, x:x + w]
    
    # Augment the cropped face
    augmented_images = augment_image(cropped_face)
    
    for augment_type, augmented_resized_feature in augmented_images:
        # Resize to desired size
        resized_face = cv2.resize(augmented_resized_feature, desired_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Construct the filename
        if len(resized_face.shape) == 2:
            # Grayscale image
            file_extension = '.jpg'
        else:
            # Color image
            file_extension = '.jpg'
        
        name = os.path.join(
            video_output_dir1,
            f"frame{frame_count}_face{face_idx}_{augment_type}{file_extension}"
        )
        
        # Save the augmented image
        if len(resized_face.shape) == 2:
            # Grayscale image saved with single channel
            cv2.imwrite(name, resized_face, [cv2.IMWRITE_JPEG_QUALITY, 100])
        else:
            # Color image
            cv2.imwrite(name, resized_face, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        print(f"Saved {name}")

# ----------------------------- Main Processing -----------------------------

def main():
    # Find all video files
    video_files = find_video_files(video_folder)
    video_files.sort()  # Sort to ensure consistent order

    total_videos = len(video_files)
    print(f"Total videos found: {total_videos}")

    if start_index >= total_videos:
        print(f"Start index {start_index} is out of range. Only {total_videos} videos found.")
        return

    # Slice the list to start from the desired video
    videos_to_process = video_files[start_index:]
    print(f"Resuming processing from video index {start_index + 1} to {total_videos}.")

    # Iterate over the videos to process
    for idx, video_path in enumerate(videos_to_process, start=start_index + 1):
        print(f"\nProcessing video {idx}/{total_videos}: {video_path}")
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(output_dir, video_name)
        video_output_dir1 = os.path.join(another_dir, video_name)
        
        # Create output directories if they don't exist
        os.makedirs(video_output_dir, exist_ok=True)
        os.makedirs(video_output_dir1, exist_ok=True)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Error] Cannot open video: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in video: {total_frames}, FPS: {fps}")

        # Determine the frame interval 'n' based on total frames
        if total_frames <= 700:
            desired_frame_count = max(30, (total_frames * 20) // 100)
            n = max(1, total_frames // desired_frame_count)
        elif 700 < total_frames <= 1400:
            desired_frame_count = max(30, (total_frames * 15) // 100)
            n = max(1, total_frames // desired_frame_count)
        else:
            desired_frame_count = max(30, (total_frames * 10) // 100)
            n = max(1, total_frames // desired_frame_count)
        
        print(f"Desired frame count: {desired_frame_count}, Frame interval: {n}")

        frame_count = 0
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            
            frame_count += 1

            # Process every 'n' frames
            if frame_count % n != 0:
                continue

            processed_frames += 1
            print(f"Processing frame {frame_count} (Processed frames: {processed_frames})")

            # Detect faces in the frame
            faces = detector(frame)
            print(f"Number of faces detected: {len(faces)}")

            for i, face in enumerate(faces):
                # Draw rectangle around the face (optional for debugging)
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Save the full face
                crop_and_save_full_face(
                    image=frame,
                    face=face,
                    video_output_dir1=video_output_dir1,
                    frame_count=frame_count,
                    face_idx=i,
                    desired_size=desired_size_full_face
                )

                # Detect facial landmarks
                shape = predictor(frame, face)
                landmarks = [(shape.part(j).x, shape.part(j).y) for j in range(68)]
                
                # Save specific facial features
                crop_resize_and_save(
                    image=frame,
                    landmarks=landmarks,
                    start_idx=18,
                    end_idx=48,
                    feature_name="full_eye",
                    video_output_dir=video_output_dir,
                    frame_count=frame_count,
                    face_idx=i,
                    desired_size=desired_size_features
                )
                crop_resize_and_save(
                    image=frame,
                    landmarks=landmarks,
                    start_idx=27,
                    end_idx=36,
                    feature_name="nose",
                    video_output_dir=video_output_dir,
                    frame_count=frame_count,
                    face_idx=i,
                    desired_size=desired_size_features
                )
                crop_resize_and_save(
                    image=frame,
                    landmarks=landmarks,
                    start_idx=48,
                    end_idx=68,
                    feature_name="mouth",
                    video_output_dir=video_output_dir,
                    frame_count=frame_count,
                    face_idx=i,
                    desired_size=desired_size_features
                )
        
            # Optionally display the frame with detected faces
            if display_window:
                cv2.imshow("Detected Faces", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[Info] Processing interrupted by user.")
                    break

        # Release the video capture object
        cap.release()
        print(f"[Info] Finished processing video: {video_path}")

    # Destroy all OpenCV windows (if any were opened)
    cv2.destroyAllWindows()
    print("\n[Info] All videos processed successfully.")

# ----------------------------- Entry Point -----------------------------

if __name__ == "__main__":
    main()
