import tensorflow as tf
import os

def load_image(path, img_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # Adjust channels if needed
    img = tf.image.resize(img, img_size)
    img = img / 255.0  # Normalize
    return img

def load_dataset(facial_dir, micro_exp_dir, label, img_size_facial, img_size_micro):
    facial_paths = [os.path.join(facial_dir, fname) for fname in sorted(os.listdir(facial_dir))]
    micro_exp_paths = [os.path.join(micro_exp_dir, fname) for fname in sorted(os.listdir(micro_exp_dir))]
    
    # Load images
    facial_images = [load_image(fp, img_size_facial) for fp in facial_paths]
    micro_exp_images = [load_image(mp, img_size_micro) for mp in micro_exp_paths]
    
    # Convert to tensor dataset
    facial_dataset = tf.data.Dataset.from_tensor_slices(facial_images)
    micro_exp_dataset = tf.data.Dataset.from_tensor_slices(micro_exp_images)
    labels_dataset = tf.data.Dataset.from_tensor_slices([label] * len(facial_images))
    
    # Zip datasets
    combined_dataset = tf.data.Dataset.zip(((facial_dataset, micro_exp_dataset), labels_dataset))
    
    return combined_dataset

def load_multiple_datasets(base_facial_dir, base_micro_exp_dir, label, img_size_facial, img_size_micro, num_folders=None):
    # Get the list of folders in the directory
    folders = sorted(os.listdir(base_facial_dir))
    
    # Limit the number of folders if specified
    if num_folders is not None:
        folders = folders[:num_folders]

    # Initialize an empty dataset
    full_dataset = tf.data.Dataset.empty()
    
    # Load each folder's dataset
    for folder in folders:
        facial_dir = os.path.join(base_facial_dir, folder)
        micro_exp_dir = os.path.join(base_micro_exp_dir, folder)
        dataset = load_dataset(facial_dir, micro_exp_dir, label, img_size_facial, img_size_micro)
        full_dataset = full_dataset.concatenate(dataset)
    
    return full_dataset

# Example usage
original_facial_dir = "Data/original/facial_frames/"
original_micro_exp_dir = "Data/original/micro_expression_frames/"
manipulated_facial_dir = "Data/manipulated/facial_frames/"
manipulated_micro_exp_dir = "Data/manipulated/micro_expression_frames/"

# Load original dataset
original_dataset = load_multiple_datasets(original_facial_dir, original_micro_exp_dir, label=0, img_size_facial=(224, 224), img_size_micro=(64, 64))

# Load manipulated dataset, limiting to the first 2190 folders
manipulated_dataset = load_multiple_datasets(manipulated_facial_dir, manipulated_micro_exp_dir, label=1, img_size_facial=(224, 224), img_size_micro=(64, 64), num_folders=2190)

# Combine and shuffle datasets
full_dataset = original_dataset.concatenate(manipulated_dataset).shuffle(buffer_size=1000).batch(32)
