# import tensorflow as tf
# import os
# from concurrent.futures import ThreadPoolExecutor
# import tqdm

# def load_image(path, img_size):
#     try:
#         img = tf.io.read_file(path)
#         img = tf.image.decode_jpeg(img, channels=3)
#         img = tf.image.resize(img, img_size)
#         img = img / 255.0  # Normalize
#         return img
#     except (tf.errors.NotFoundError, tf.errors.InvalidArgumentError, tf.errors.UnknownError, OSError) as e:
#         print(f"Skipping file {path} due to error: {e}")
#         return None

# def parallel_load_images(image_paths, img_size):
#     def load_and_resize(path):
#         img = load_image(path, img_size)
#         if img is None:
#             return None
#         return img
    
#     with ThreadPoolExecutor(max_workers=8) as executor:
#         images = list(tqdm.tqdm(executor.map(load_and_resize, image_paths), total=len(image_paths), desc="Loading images"))
    
#     # Filter out any None values (failed loads)
#     images = [img for img in images if img is not None]
#     return images

# def load_dataset(facial_dir, micro_exp_dir, label, img_size_facial, img_size_micro):
#     try:
#         facial_paths = [os.path.join(facial_dir, fname) for fname in sorted(os.listdir(facial_dir))]
#         micro_exp_paths = [os.path.join(micro_exp_dir, fname) for fname in sorted(os.listdir(micro_exp_dir))]
#     except (FileNotFoundError, PermissionError, OSError) as e:
#         print(f"Skipping folder {facial_dir} or {micro_exp_dir} due to error: {e}")
#         return tf.data.Dataset.from_tensor_slices(((), (), ()))  # Return an empty dataset if folder is missing

#     # Load images with multithreading
#     facial_images = parallel_load_images(facial_paths, img_size_facial)
#     micro_exp_images = parallel_load_images(micro_exp_paths, img_size_micro)

#     # Ensure both lists are of the same length by taking the minimum length
#     min_length = min(len(facial_images), len(micro_exp_images))
#     facial_images, micro_exp_images = facial_images[:min_length], micro_exp_images[:min_length]

#     # Convert to tensor dataset
#     facial_dataset = tf.data.Dataset.from_tensor_slices(facial_images)
#     micro_exp_dataset = tf.data.Dataset.from_tensor_slices(micro_exp_images)
#     labels_dataset = tf.data.Dataset.from_tensor_slices([label] * len(facial_images))
    
#     combined_dataset = tf.data.Dataset.zip(((facial_dataset, micro_exp_dataset), labels_dataset))
    
#     return combined_dataset

# def load_multiple_datasets(base_facial_dir, base_micro_exp_dir, label, img_size_facial, img_size_micro):
#     folders = sorted(os.listdir(base_facial_dir))

#     full_dataset = tf.data.Dataset.from_tensor_slices(((tf.zeros((0, 224, 224, 3)), tf.zeros((0, 64, 64, 3))), tf.zeros((0,), dtype=tf.int32)))
    
#     for folder in tqdm.tqdm(folders, desc="Processing folders"):
#         facial_dir = os.path.join(base_facial_dir, folder)
#         micro_exp_dir = os.path.join(base_micro_exp_dir, folder)

#         # Attempt to load the dataset for each folder
#         dataset = load_dataset(facial_dir, micro_exp_dir, label, img_size_facial, img_size_micro)
#         full_dataset = full_dataset.concatenate(dataset)
    
#     return full_dataset

# def create_full_dataset(original_facial_dir, original_micro_exp_dir, manipulated_facial_dir, manipulated_micro_exp_dir, img_size_facial=(224, 224), img_size_micro=(64, 64)):
#     original_dataset = load_multiple_datasets(original_facial_dir, original_micro_exp_dir, label=0, img_size_facial=img_size_facial, img_size_micro=img_size_micro)
#     manipulated_dataset = load_multiple_datasets(manipulated_facial_dir, manipulated_micro_exp_dir, label=1, img_size_facial=img_size_facial, img_size_micro=img_size_micro)

#     full_dataset = original_dataset.concatenate(manipulated_dataset).shuffle(buffer_size=1000).batch(16).prefetch(tf.data.AUTOTUNE)
    
#     return full_dataset

# # Define paths to the original and manipulated data directories
# base_data_dir = 'D:/Projects/Face-Swap-Detection-Model/Preprocessed_data'

# original_facial_dir = os.path.join(base_data_dir, 'original', 'facial_frames')
# original_micro_exp_dir = os.path.join(base_data_dir, 'original', 'micro_expression_frames')

# manipulated_facial_dir = os.path.join(base_data_dir, 'manipulated', 'facial_frames')
# manipulated_micro_exp_dir = os.path.join(base_data_dir, 'manipulated', 'micro_expression_frames')

# # Load the dataset
# full_dataset = create_full_dataset(
#     original_facial_dir=original_facial_dir,
#     original_micro_exp_dir=original_micro_exp_dir,
#     manipulated_facial_dir=manipulated_facial_dir,
#     manipulated_micro_exp_dir=manipulated_micro_exp_dir,
#     img_size_facial=(224, 224),
#     img_size_micro=(64, 64)
# )

# # Iterate over the dataset (for example, to check batch shapes)
# for batch in full_dataset.take(1):
#     (facial_images, micro_exp_images), labels = batch
#     print("Facial Images Batch Shape:", facial_images.shape)
#     print("Micro-expression Images Batch Shape:", micro_exp_images.shape)
#     print("Labels Batch Shape:", labels.shape)

# import tensorflow as tf

# def serialize_example(facial_image, micro_image, label):
#     # Convert individual data types to bytes, int64, or float as required for TFRecord format
#     feature = {
#         'facial_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(tf.image.convert_image_dtype(facial_image, tf.uint8)).numpy()])),
#         'micro_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(tf.image.convert_image_dtype(micro_image, tf.uint8)).numpy()])),
#         'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
#     }
    
#     # Create a Features message using tf.train.Example
#     example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
#     return example_proto.SerializeToString()

# def save_dataset_to_tfrecord(dataset, tfrecord_file_path):
#     with tf.io.TFRecordWriter(tfrecord_file_path) as writer:
#         for (facial_image, micro_image), label in dataset:
#             for i in range(len(label)):
#                 serialized_example = serialize_example(facial_image[i], micro_image[i], label[i].numpy())
#                 writer.write(serialized_example)

#     print(f"Dataset successfully saved to {tfrecord_file_path}")

# # Usage
# tfrecord_file_path = "D:/Projects/Face-Swap-Detection-Model/dataset.tfrecord"
# save_dataset_to_tfrecord(full_dataset, tfrecord_file_path)


# import os
# from PIL import Image
# import pickle
# from io import BytesIO
# from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm

# # Function to convert PIL images to byte arrays
# def pil_to_bytes(pil_img):
#     with BytesIO() as buffer:
#         pil_img.save(buffer, format='JPEG')
#         return buffer.getvalue()

# # Helper function to process a single frame file
# def process_frame(frame_path, data_type, label):
#     try:
#         with Image.open(frame_path) as img:
#             # Convert the image to a byte array (JPEG)
#             byte_data = pil_to_bytes(img)
#             if data_type == 'facial':
#                 return ('frames', byte_data, label)
#             elif data_type == 'micro_expression':
#                 return ('Micro_Expression', byte_data, label)
#     except Exception as e:
#         print(f"Skipping file due to error: {frame_path}, Error: {e}")
#         return None  # Return None if there's an issue

# # Main function to load and save data with multithreading and progress tracking
# def load_and_save_data(base_path, save_path='D:/Projects/Face-Swap-Detection-Model/video_data_DF_manipulated_001.pkl', max_workers=4):
#     main_folders = ['original', 'manipulated']
#     data_types = ['facial', 'micro_expression']

#     # Initialize video data structure
#     video_data = {}

#     # Calculate total steps for the overall progress bar
#     total_steps = sum(
#         len(os.listdir(os.path.join(base_path, main_folder, data_type)))
#         for main_folder in main_folders
#         for data_type in data_types
#     )

#     # Overall progress bar
#     with tqdm(total=total_steps, desc="Total Progress", mininterval=1.0) as overall_progress:
#         # Iterate through each main folder
#         for main_folder in main_folders:
#             label = 0 if main_folder == 'original' else 1
#             folder_path = os.path.join(base_path, main_folder)

#             # Iterate through data types (facial frames and micro expressions)
#             for data_type in data_types:
#                 data_type_path = os.path.join(folder_path, data_type)

#                 # List of all video folders in the current data type path
#                 video_folders = [f for f in os.listdir(data_type_path) if os.path.isdir(os.path.join(data_type_path, f))]

#                 # Process each video folder
#                 for video_folder in video_folders:
#                     video_folder_path = os.path.join(data_type_path, video_folder)

#                     # Initialize video entry in the dictionary if not already present
#                     if video_folder not in video_data:
#                         video_data[video_folder] = {
#                             'frames': [],
#                             'frames_label': [],  # Corrected label key
#                             'Micro_Expression': [],
#                             'Micro_Expression_label': []
#                         }

#                     frame_files = os.listdir(video_folder_path)

#                     # Use multithreading to process frame files in parallel
#                     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#                         # Submitting tasks for each frame file
#                         futures = [executor.submit(process_frame, os.path.join(video_folder_path, frame_file), data_type, label) for frame_file in frame_files]

#                         # Processing the results as they complete
#                         for future in futures:
#                             result = future.result()
#                             if result:
#                                 data_key, byte_data, lbl = result
#                                 video_data[video_folder][data_key].append(byte_data)
#                                 video_data[video_folder][f"{data_key}_label"].append(lbl)
#                             overall_progress.update(1)  # Update the overall progress bar for each frame processed

#     # Save the dataset to a pickle file
#     with open(save_path, 'wb') as f:
#         pickle.dump(video_data, f)

#     print("Data loading and saving completed.")

# # Example usage
# load_and_save_data("G:/Deepfakes/manipulated")



#** Alternative Main Function to Load and Save Data with Multithreading and Progress Tracking**

# import os
# from PIL import Image
# import pickle
# from io import BytesIO
# from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm
# import aiofiles
# import asyncio

# # Function to convert PIL images to byte arrays
# def pil_to_bytes(pil_img):
#     with BytesIO() as buffer:
#         pil_img.save(buffer, format='JPEG')
#         return buffer.getvalue()

# # Helper function to process a single frame file
# def process_frame(frame_path, data_type, label):
#     try:
#         with Image.open(frame_path) as img:
#             # Convert the image to a byte array (JPEG)
#             byte_data = pil_to_bytes(img)
#             if data_type == 'facial':
#                 return ('frames', byte_data, label)
#             elif data_type == 'micro_expression':
#                 return ('Micro_Expression', byte_data, label)
#     except Exception as e:
#         print(f"Skipping file due to error: {frame_path}, Error: {e}")
#         return None  # Return None if there's an issue

# # Asynchronous save function to write to a pickle file
# async def async_save_data(video_data, save_path):
#     async with aiofiles.open(save_path, 'wb') as f:
#         await f.write(pickle.dumps(video_data))
#     print(f"Partial data successfully saved to {save_path}")

# # Main function to load and save data with multithreading, batch processing, and async I/O
# def load_and_save_data(base_path, save_path='D:/Projects/Face-Swap-Detection-Model/video_data_DF_001.pkl', max_workers=4, batch_size=500):
#     main_folders = ['original', 'manipulated']
#     data_types = ['facial', 'micro_expression']
#     video_data = {}
#     batch_counter = 0  # Counter for periodic saves

#     for main_folder in tqdm(main_folders, desc="Main Folders", mininterval=1.0):
#         label = 0 if main_folder == 'original' else 1
#         folder_path = os.path.join(base_path, main_folder)

#         for data_type in tqdm(data_types, desc=f"{main_folder} - Data Types", leave=False, mininterval=1.0):
#             data_type_path = os.path.join(folder_path, data_type)
#             video_folders = [f for f in os.listdir(data_type_path) if os.path.isdir(os.path.join(data_type_path, f))]

#             for video_folder in tqdm(video_folders, desc=f"{data_type} - Video Folders", leave=False, mininterval=1.0):
#                 video_folder_path = os.path.join(data_type_path, video_folder)

#                 if video_folder not in video_data:
#                     video_data[video_folder] = {
#                         'frames': [],
#                         'frames_label': [],
#                         'Micro_Expression': [],
#                         'Micro_Expression_label': []
#                     }

#                 frame_files = os.listdir(video_folder_path)
#                 with ThreadPoolExecutor(max_workers=max_workers) as executor:
#                     futures = [
#                         executor.submit(process_frame, os.path.join(video_folder_path, frame_file), data_type, label)
#                         for frame_file in frame_files
#                     ]

#                     for future in tqdm(futures, desc="Frames", leave=False, mininterval=1.0):
#                         result = future.result()
#                         if result:
#                             data_key, byte_data, lbl = result
#                             video_data[video_folder][data_key].append(byte_data)
#                             video_data[video_folder][f"{data_key}_label"].append(lbl)

#                 batch_counter += 1

#                 # Periodically save to pickle file
#                 if batch_counter % batch_size == 0:
#                     asyncio.run(async_save_data(video_data, save_path))
#                     video_data.clear()  # Clear data to free memory after saving

#     # Final save for any remaining data
#     asyncio.run(async_save_data(video_data, save_path))
#     print("Data loading and saving completed.")

# # Example usage
# load_and_save_data("G:/Deepfakes")


#* Fully changed code

import os
from PIL import Image
import pickle
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import math

# Function to convert PIL images to byte arrays
def pil_to_bytes(pil_img):
    with BytesIO() as buffer:
        pil_img.save(buffer, format='JPEG')
        return buffer.getvalue()

# Helper function to process a single frame file
def process_frame(frame_path, data_type, label):
    try:
        with Image.open(frame_path) as img:
            byte_data = pil_to_bytes(img)
            if data_type == 'facial':
                return ('frames', byte_data, label)
            elif data_type == 'micro_expression':
                return ('Micro_Expression', byte_data, label)
    except Exception as e:
        print(f"Skipping file due to error: {frame_path}, Error: {e}")
        return None  # Return None if there's an issue

# Main function to load, process, and save data in batches
def load_and_save_data_in_batches(base_path, folder_to_process='manipulated', save_dir='D:/Projects/Face-Swap-Detection-Model', max_workers=4, batch_size=500):
    data_types = ['facial', 'micro_expression']
    label = 0 if folder_to_process == 'original' else 1
    folder_path = os.path.join(base_path, folder_to_process)

    # List all video folders
    all_video_folders = [f for f in os.listdir(os.path.join(folder_path, data_types[0])) if os.path.isdir(os.path.join(folder_path, data_types[0], f))]
    total_batches = math.ceil(len(all_video_folders) / batch_size)

    # Process each batch
    for batch_idx in range(total_batches):
        video_data = {}
        batch_folders = all_video_folders[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        # Progress bar for the batch
        with tqdm(total=len(batch_folders) * len(data_types), desc=f"Processing Batch {batch_idx + 1}/{total_batches}", mininterval=3.0) as batch_progress:
            # Process each video folder in the batch
            for video_folder in batch_folders:
                video_data[video_folder] = {
                    'frames': [],
                    'frames_label': [],
                    'Micro_Expression': [],
                    'Micro_Expression_label': []
                }

                # Process each data type for the video
                for data_type in data_types:
                    data_type_path = os.path.join(folder_path, data_type, video_folder)
                    frame_files = os.listdir(data_type_path)

                    # Multithreading for processing frame files in parallel
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [executor.submit(process_frame, os.path.join(data_type_path, frame_file), data_type, label) for frame_file in frame_files]
                        for future in futures:
                            result = future.result()
                            if result:
                                data_key, byte_data, lbl = result
                                video_data[video_folder][data_key].append(byte_data)
                                video_data[video_folder][f"{data_key}_label"].append(lbl)
                    batch_progress.update(1)  # Update progress for each video folder processed

        # Save the current batch to a pickle file
        save_path = os.path.join(save_dir, f'{folder_to_process}_video_data_batch_{batch_idx + 1}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(video_data, f)

        print(f"Batch {batch_idx + 1} saved to {save_path}")

        # Clear video_data for the next batch
        video_data.clear()

    print("All batches processed and saved.")

# Example usage:
load_and_save_data_in_batches("G:/Deepfakes", folder_to_process='manipulated', save_dir='D:/Projects/Face-Swap-Detection-Model')

load_and_save_data_in_batches("G:/Deepfakes", folder_to_process='original', save_dir='D:/Projects/Face-Swap-Detection-Model')



