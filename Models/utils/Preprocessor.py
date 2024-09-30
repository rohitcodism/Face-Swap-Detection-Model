import cv2
import dlib
import os
import random
import numpy as np

# Directory setup for parts of face
output_dir = "main_directory"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Directory setup for whole face
another_dir = "face_directory"
if not os.path.exists(another_dir):
    os.makedirs(another_dir)

video_folder = "Data_big"

# Load Dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Pre-trained model

# Desired size for resizing the features (optional)
desired_size = (64, 64)

# Maintaining Aspect Ratio for better quality
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
    noisy_image = image + gauss.reshape(row, col, ch)
    noisy_image = np.clip(noisy_image, 0, 255).astype('uint8')
    return noisy_image

# Function to convert image to grayscale
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Augmentation function
def augment_image(image):
    augmented_images = []
    
    # Original
    augmented_images.append(("original", image))
    
    # Random horizontal and vertical flips
   
    image_hor = cv2.flip(image, 1)
    augmented_images.append(("flipped_horizontal", image_hor))
        
    image_ver = cv2.flip(image, 0)
    augmented_images.append(("flipped_vertical", image_ver))
    
    # Rotation by -45 and +45 degrees
    h, w = image.shape[:2]
    M_left = cv2.getRotationMatrix2D((w // 2, h // 2), -45, 1)
    rotate_left = cv2.warpAffine(image, M_left, (w, h))
    augmented_images.append(("rotated_left", rotate_left))
    
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

# Function to resize while maintaining aspect ratio and pad the image to the desired size
def crop_resize_and_save(image, landmarks, start_idx, end_idx, feature_name, video_output_dir, frame_count, i, desired_size):
    points = landmarks[start_idx:end_idx]
    
    x, y, w, h = cv2.boundingRect(np.array(points))

    # Adjust invalid bounding box dimensions
    if w <= 0 or h <= 0:
        print(f"Adjusting {feature_name} bounding box for frame {frame_count} face {i}: Invalid w={w}, h={h}")
        w = max(w, 30)
        h = max(h, 30)

    if x < 0: x = 0
    if y < 0: y = 0
    if x + w > image.shape[1]: w = image.shape[1] - x
    if y + h > image.shape[0]: h = image.shape[0] - y

    cropped_feature = image[y:y + h, x:x + w]

    # Handle case where cropping results in an empty image
    if cropped_feature.size == 0:
        print(f"Empty crop for {feature_name}, using whole face for frame {frame_count} face {i}")
        cropped_feature = image

    # Maintain aspect ratio while resizing
    h, w = cropped_feature.shape[:2]
    aspect_ratio = w / h

    # Compute new dimensions while maintaining aspect ratio
    if aspect_ratio > 1:  # Width is greater than height
        new_w = desired_size[0]
        new_h = int(new_w / aspect_ratio)
    else:  # Height is greater than width
        new_h = desired_size[1]
        new_w = int(new_h * aspect_ratio)

    resized_feature = cv2.resize(cropped_feature, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Create a blank image of the desired size and center the resized feature on it
    
    padded_feature = np.full((desired_size[1], desired_size[0], 3), 0, dtype=np.uint8)  # Create a white background

    # Compute padding to center the image
    x_offset = (desired_size[0] - new_w) // 2
    y_offset = (desired_size[1] - new_h) // 2

    # Paste the resized feature into the padded image
    padded_feature[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_feature

    # Augment the resized and padded image
    
    augmented_images = augment_image(padded_feature)
    
    
    for augment_type, augmented_resized_feature in augmented_images:
        
        normalized_face = augmented_resized_feature.astype("float32") / 255.0

        #cv2.imshow(f"Normalized Face {i}", normalized_face)
    
        resized_name = os.path.join(video_output_dir, f"frame{frame_count}face{i}{feature_name}_{augment_type}.jpg")
        cv2.imwrite(resized_name, normalized_face, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(f"Saved {resized_name}")




def crop_and_save_full_face(image, face, video_output_dir1, frame_count, i, desired_size):
    x, y, w, h = face.left(), face.top(), face.width(), face.height()

    if x < 0: x = 0
    if y < 0: y = 0
    if x + w > image.shape[1]: w = image.shape[1] - x
    if y + h > image.shape[0]: h = image.shape[0] - y

    cropped_face = image[y:y + h, x:x + w]
    
    augmented_images = augment_image(cropped_face)
    for augment_type, augmented_resized_feature in augmented_images:
        
        # Normalize the augmented image
        normalized_face = augmented_resized_feature.astype("float32") / 255.0
        
        cv2.imshow(f"Normalized Face {i}", normalized_face)
        
        resized_face = cv2.resize(normalized_face, desired_size, interpolation=cv2.INTER_LANCZOS4)
        name = os.path.join(video_output_dir1, f"frame{frame_count}face{i}{augment_type}.jpg")
        cv2.imwrite(name, resized_face, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(f"Saved {name}")

# Main loop to process videos
def find_video_files(root_folder):
    video_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_files.append(os.path.join(root, file))
    return video_files

# Process videos
video_files = find_video_files(video_folder)
for video_path in video_files:
    print(f"Processing video: {video_path}")
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    video_output_dir1 = os.path.join(another_dir, video_name)
    
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)
    if not os.path.exists(video_output_dir1):
        os.makedirs(video_output_dir1)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    
    if total_frames <= 700:
        desired_frame_count = max(30, (total_frames * 20) // 100)
        n = max(1, total_frames // desired_frame_count)
    elif 700 < total_frames <= 1400:
        desired_frame_count = max(30, (total_frames * 15) // 100)
        n = max(1, total_frames // desired_frame_count)
    else:
        desired_frame_count = max(30, (total_frames * 10) // 100)
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
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # crop_and_save_full_face(frame, face, video_output_dir1, frame)
            crop_and_save_full_face(frame, face, video_output_dir1, frame_count, i, desired_size=(224, 224))
            shape = predictor(frame, face)
            landmarks = [(shape.part(j).x, shape.part(j).y) for j in range(68)]
            
            crop_resize_and_save(frame, landmarks, 18, 48, "full_eye", video_output_dir, frame_count, i, desired_size)
            crop_resize_and_save(frame, landmarks, 27, 36, "nose", video_output_dir, frame_count, i, desired_size)
            crop_resize_and_save(frame, landmarks, 48, 68, "mouth", video_output_dir, frame_count, i, desired_size)
                
        cv2.imshow("Detected Faces", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
cv2.destroyAllWindows()