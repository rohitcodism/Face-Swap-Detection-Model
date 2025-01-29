import tensorflow as tf
import pickle
import os

def serialize_example(frame, micro_exp, label):
    feature = {
        'frame': tf.train.Feature(bytes_list=tf.train.BytesList(value=[frame])),
        'micro_exp': tf.train.Feature(bytes_list=tf.train.BytesList(value=[micro_exp])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def create_tfrecord_from_pickle(pickle_path, tfrecord_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
        
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for video_name, video_info in data.items():
            for i in range(len(video_info['frames'])):
                frame = video_info['frames'][i]
                micro_exp = video_info['Micro_Expression'][i]
                label = video_info['frames_label'][i]
                example = serialize_example(frame, micro_exp, label)
                writer.write(example)

# Convert each .pkl batch to .tfrecord
pickle_folder = 'D:/Projects/Face-Swap-Detection-Model/DF'
tfrecord_folder = 'D:/Projects/Face-Swap-Detection-TFRecords/DF'
os.makedirs(tfrecord_folder, exist_ok=True)

for file_name in os.listdir(pickle_folder):
    if file_name.endswith('.pkl'):
        pickle_path = os.path.join(pickle_folder, file_name)
        tfrecord_path = os.path.join(tfrecord_folder, file_name.replace('.pkl', '.tfrecord'))
        create_tfrecord_from_pickle(pickle_path, tfrecord_path)


def parse_tfrecord_fn(example):
    feature_description = {
        'frame': tf.io.FixedLenFeature([], tf.string),
        'micro_exp': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)

    # Decode and ensure the frame is RGB
    frame = tf.io.decode_jpeg(example['frame'])  # Assuming images are stored in JPEG format
    frame = tf.image.convert_image_dtype(frame, dtype=tf.float32)  # Convert to float32 if needed
    if frame.shape[-1] == 1:
        frame = tf.image.grayscale_to_rgb(frame)

    # Decode and ensure the micro-expression is RGB
    micro_exp = tf.io.decode_jpeg(example['micro_exp'])
    micro_exp = tf.image.convert_image_dtype(micro_exp, dtype=tf.float32)  # Convert to float32 if needed
    if micro_exp.shape[-1] == 1:
        micro_exp = tf.image.grayscale_to_rgb(micro_exp)

    label = example['label']
    return (frame, micro_exp), label


# Create a Dataset from TFRecords
def create_dataset(tfrecord_folder, batch_size=16):
    tfrecord_files = [os.path.join(tfrecord_folder, f) for f in os.listdir(tfrecord_folder) if f.endswith('.tfrecord')]
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

import matplotlib.pyplot as plt
import tensorflow as tf

# Verify a few samples from the TFRecord dataset
def inspect_tfrecord_data(dataset, num_batches=1):
    for batch_idx, (frames, labels) in enumerate(dataset.take(num_batches)):
        print(f"Batch {batch_idx + 1}:")

        # Unpack frames and micro_expressions
        frame_imgs, micro_exp_imgs = frames
        for i in range(len(frame_imgs)):
            print(f"Sample {i + 1}:")
            print(f" - Frame shape: {frame_imgs[i].shape}")
            print(f" - Micro-Expression shape: {micro_exp_imgs[i].shape}")
            print(f" - Label: {labels[i].numpy()}")

            # Plot one frame and micro expression per sample for visual inspection
            plt.figure(figsize=(8, 4))
            
            # Display the main frame image
            plt.subplot(1, 2, 1)
            plt.imshow(tf.image.convert_image_dtype(frame_imgs[i], dtype=tf.uint8))
            plt.title("Frame")
            
            # Display the micro expression image
            plt.subplot(1, 2, 2)
            plt.imshow(tf.image.convert_image_dtype(micro_exp_imgs[i], dtype=tf.uint8))
            plt.title("Micro Expression")
            
            plt.show()