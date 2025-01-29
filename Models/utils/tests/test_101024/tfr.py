import tensorflow as tf
import os
import concurrent.futures
from tqdm import tqdm

# Function to write a single TFRecord file
def write_tfrecord(file_name, data):
    with tf.io.TFRecordWriter(file_name) as writer:
        for x_frames, x_micro_exp, label in data:
            example = serialize_example(x_frames, x_micro_exp, label)
            writer.write(example)

# Update the main function to use multithreading and progress tracking
def save_dataset_multithreaded(output_dir, data_generator, num_threads=4):
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize a progress bar
    total_batches = sum(1 for _ in data_generator)  # Count total batches
    data_generator = create_full_dataset(original_facial_dir, original_micro_exp_dir, manipulated_facial_dir, manipulated_micro_exp_dir) # create data generator for tracking
    progress_bar = tqdm(total=total_batches, desc="Saving TFRecords", unit="batch")

    # Create a thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i, batch in enumerate(data_generator):
            # Generate a unique file name for each batch
            file_name = os.path.join(output_dir, f"data_batch_{i}.tfrecord")
            future = executor.submit(write_tfrecord, file_name, batch)
            futures.append(future)

            # Update progress bar for each submitted task
            future.add_done_callback(lambda p: progress_bar.update(1))

        # Wait for all threads to complete
        concurrent.futures.wait(futures)
    
    progress_bar.close()  # Close the progress bar when done

# Usage example
output_dir = "G:/Preprocessed_data/tfrecords/"
save_dataset_multithreaded(output_dir, full_dataset)
