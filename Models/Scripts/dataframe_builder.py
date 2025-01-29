import os
import tensorflow as tf

def load_dataset(facial_dir, micro_exp_dir, label, img_size_facial, img_size_micro):
    """
    Load facial and micro-expression frames as TensorFlow datasets.
    
    Args:
    - facial_dir (str): Directory containing facial frames.
    - micro_exp_dir (str): Directory containing micro-expression frames.
    - label (int): Label for the dataset (0 for original, 1 for manipulated).
    - img_size_facial (tuple): Target size for facial frames.
    - img_size_micro (tuple): Target size for micro-expression frames.
    
    Returns:
    - tf.data.Dataset: A TensorFlow dataset containing image data and labels.
    """
    try:
        # Check and get all paths from the directories
        facial_paths = [os.path.join(facial_dir, fname) for fname in sorted(os.listdir(facial_dir))]
        micro_exp_paths = [os.path.join(micro_exp_dir, fname) for fname in sorted(os.listdir(micro_exp_dir))]
        
        # Ensure that there are files to process
        if not facial_paths or not micro_exp_paths:
            print(f"Skipping empty directory: {facial_dir} or {micro_exp_dir}")
            return tf.data.Dataset.from_tensor_slices(([], [], []))

    except (FileNotFoundError, PermissionError, OSError) as e:
        print(f"Skipping folder {facial_dir} or {micro_exp_dir} due to error: {e}")
        return tf.data.Dataset.from_tensor_slices(([], [], []))  # Return an empty dataset if folder is missing

    # Load images with multithreading
    facial_images = parallel_load_images(facial_paths, img_size_facial)
    micro_exp_images = parallel_load_images(micro_exp_paths, img_size_micro)
    
    # Return dataset with labels
    labels = tf.constant(label, shape=(len(facial_images),))
    return tf.data.Dataset.from_tensor_slices((facial_images, micro_exp_images, labels))


def parallel_load_images(paths, target_size):
    """
    Helper function to load images in parallel.
    
    Args:
    - paths (list): List of image paths.
    - target_size (tuple): Target size to resize images.
    
    Returns:
    - list: List of loaded and resized images.
    """
    images = []
    for path in paths:
        img = tf.keras.preprocessing.image.load_img(path, target_size=target_size)
        img = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img)
    return images


def load_multiple_datasets(base_facial_dir, base_micro_exp_dir, label, img_size_facial, img_size_micro, num_folders=None):
    """
    Load multiple datasets from specified directories.

    Args:
    - base_facial_dir (str): Base directory for facial frames.
    - base_micro_exp_dir (str): Base directory for micro-expression frames.
    - label (int): Label for the dataset.
    - img_size_facial (tuple): Target size for facial frames.
    - img_size_micro (tuple): Target size for micro-expression frames.
    - num_folders (int): Limit on the number of folders to process.

    Returns:
    - tf.data.Dataset: Concatenated dataset from all folders.
    """
    folders = sorted(os.listdir(base_facial_dir))[:num_folders]
    full_dataset = tf.data.Dataset.from_tensor_slices(([], [], []))

    for folder in folders:
        facial_dir = os.path.join(base_facial_dir, folder)
        micro_exp_dir = os.path.join(base_micro_exp_dir, folder)

        # Load dataset for each folder
        dataset = load_dataset(facial_dir, micro_exp_dir, label, img_size_facial, img_size_micro)
        full_dataset = full_dataset.concatenate(dataset)

    return full_dataset


def create_full_dataset(original_facial_dir, original_micro_exp_dir, manipulated_facial_dir, manipulated_micro_exp_dir, 
                        img_size_facial=(224, 224), img_size_micro=(64, 64), num_manipulated_folders=2190):
    """
    Create a complete dataset by combining original and manipulated datasets.

    Args:
    - original_facial_dir (str): Directory for original facial frames.
    - original_micro_exp_dir (str): Directory for original micro-expression frames.
    - manipulated_facial_dir (str): Directory for manipulated facial frames.
    - manipulated_micro_exp_dir (str): Directory for manipulated micro-expression frames.
    - img_size_facial (tuple): Size for facial frames.
    - img_size_micro (tuple): Size for micro-expression frames.
    - num_manipulated_folders (int): Number of manipulated folders to process.

    Returns:
    - tf.data.Dataset: The full dataset with original and manipulated data.
    """
    original_dataset = load_multiple_datasets(original_facial_dir, original_micro_exp_dir, label=0, 
                                              img_size_facial=img_size_facial, img_size_micro=img_size_micro)
    manipulated_dataset = load_multiple_datasets(manipulated_facial_dir, manipulated_micro_exp_dir, label=1, 
                                                 img_size_facial=img_size_facial, img_size_micro=img_size_micro, 
                                                 num_folders=num_manipulated_folders)
    
    full_dataset = original_dataset.concatenate(manipulated_dataset).shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)
    return full_dataset

# Directories
original_facial_dir = '/Volumes/New Volume/Preprocessed_data/original/facial_frames'
original_micro_exp_dir = '/Volumes/New Volume/Preprocessed_data/original/micro_expression_frames'
manipulated_facial_dir = '/Volumes/New Volume/Preprocessed_data/manipulated/facial_frames'
manipulated_micro_exp_dir = '/Volumes/New Volume/Preprocessed_data/manipulated/micro_expression_frames'

# Create the full dataset
full_dataset = create_full_dataset(original_facial_dir, original_micro_exp_dir, manipulated_facial_dir, manipulated_micro_exp_dir)

# Check dataset and progress
for batch in full_dataset.take(1):
    print(batch)
