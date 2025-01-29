import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
from io import BytesIO
import os
import pickle

# Function to load data from multiple .pkl files
def load_data_from_pickles(pickle_dir):
    data = {}
    # List all .pkl files in the directory
    pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.pkl')]
    
    # Load data from each .pkl file and merge into a single dictionary
    for pickle_file in pickle_files:
        with open(os.path.join(pickle_dir, pickle_file), 'rb') as f:
            batch_data = pickle.load(f)
            data.update(batch_data)
    
    return data

def load_batch_data_from_pickles(pickle_dir, batch_size=16):
    """
    Load data from pickle files in batches to avoid memory overload.
    """
    # List all .pkl files in the directory
    pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.pkl')]

    for pickle_file in pickle_files:
        # Load data from the pickle file
        with open(os.path.join(pickle_dir, pickle_file), 'rb') as f:
            batch_data = pickle.load(f)
        
        # Yield the batch data one batch at a time to avoid overloading memory
        data_keys = list(batch_data.keys())
        for i in range(0, len(data_keys), batch_size):
            batch = {key: batch_data[key] for key in data_keys[i:i + batch_size]}
            yield batch

class VideoDataGenerator(Sequence):
    def __init__(self, data, batch_size=16, img_height=224, img_width=224, micro_exp_height=64, micro_exp_width=64):
        self.data = data
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.micro_exp_height = micro_exp_height
        self.micro_exp_width = micro_exp_width
        self.video_names = list(data.keys())
        self.indexes = np.arange(len(self.video_names))

    def __len__(self):
        return int(np.ceil(len(self.video_names) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_video_names = [self.video_names[i] for i in indexes]
        X_frames = []
        X_micro_exp = []
        y = []

        for video_name in batch_video_names:
            video_info = self.data[video_name]
            
            # Check if frames and micro expressions lists are not empty
            if not video_info['frames'] or not video_info['Micro_Expression']:
                print(f"Skipping {video_name} due to empty frames or micro expressions.")
                continue
            
            # Load the first frame and micro expression as images
            frame = Image.open(BytesIO(video_info['frames'][0])).resize((self.img_width, self.img_height))
            micro_exp = Image.open(BytesIO(video_info['Micro_Expression'][0])).resize((self.micro_exp_width, self.micro_exp_height))
            
            # Convert to numpy arrays
            frame_np = np.array(frame)
            micro_exp_np = np.array(micro_exp)
            
            # Normalize the frames and micro expressions by dividing by 255.0
            frame_np = frame_np / 255.0
            micro_exp_np = micro_exp_np / 255.0
            
            X_frames.append(frame_np)
            X_micro_exp.append(micro_exp_np)
            y.append(video_info['frame_label'][0])  # Adjust if needed for multiple labels
        
        # Convert to numpy arrays (ensure correct dtype)
        X_frames = np.array(X_frames, dtype=np.float32)
        X_micro_exp = np.array(X_micro_exp, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        y = y.reshape(-1, 1)  # reshape y to (batch_size, 1)
        
        return ((X_frames, X_micro_exp), y)

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)


import numpy as np
from tensorflow.keras.utils import Sequence
from PIL import Image
from io import BytesIO
import os
import pickle

class VideoDataGenerator2(Sequence):
    def __init__(self, pickle_dir, batch_size=16, img_height=224, img_width=224, micro_exp_height=64, micro_exp_width=64):
        self.pickle_dir = pickle_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.micro_exp_height = micro_exp_height
        self.micro_exp_width = micro_exp_width

        # Load the data from pickle files
        self.data = self.load_data_from_pickles()

        # Separate original and manipulated data
        self.original_data = {k: v for k, v in self.data.items() if v['frames_label'][0] == 0}
        self.manipulated_data = {k: v for k, v in self.data.items() if v['frames_label'][0] == 1}

        # Calculate original and manipulated video lists
        self.original_videos = list(self.original_data.keys())
        self.manipulated_videos = list(self.manipulated_data.keys())

        # Oversample original videos to exactly match the number of manipulated videos
        oversample_count = len(self.manipulated_videos)
        self.original_videos = np.tile(self.original_videos, (oversample_count // len(self.original_videos)) + 1)
        self.original_videos = self.original_videos[:oversample_count]  # Trim excess if any

        # Shuffle the videos
        np.random.shuffle(self.original_videos)
        np.random.shuffle(self.manipulated_videos)

    def load_data_from_pickles(self):
        data = {}
        pickle_files = [f for f in os.listdir(self.pickle_dir) if f.endswith('.pkl')]
        
        for pickle_file in pickle_files:
            with open(os.path.join(self.pickle_dir, pickle_file), 'rb') as f:
                batch_data = pickle.load(f)
                data.update(batch_data)
        
        return data

    def __len__(self):
        # Number of batches per epoch
        return int(np.ceil(len(self.original_videos) / self.batch_size / 2))  # 50% original, 50% manipulated

    def __getitem__(self, index):
        # Load a batch of data, half from original, half from manipulated videos
        original_batch = self.original_videos[index * self.batch_size // 2:(index + 1) * self.batch_size // 2]
        manipulated_batch = self.manipulated_videos[index * self.batch_size // 2:(index + 1) * self.batch_size // 2]

        X_frames = []
        X_micro_exp = []
        y = []

        # Process original videos
        for video_name in original_batch:
            video_info = self.data[video_name]

            if not video_info['frames'] or not video_info['Micro_Expression']:
                continue
            
            frame = Image.open(BytesIO(video_info['frames'][0])).resize((self.img_width, self.img_height))
            micro_exp = Image.open(BytesIO(video_info['Micro_Expression'][0])).resize((self.micro_exp_width, self.micro_exp_height))

            X_frames.append(np.array(frame))
            X_micro_exp.append(np.array(micro_exp))
            y.append(video_info['frames_label'][0])

        # Process manipulated videos
        for video_name in manipulated_batch:
            video_info = self.data[video_name]

            if not video_info['frames'] or not video_info['Micro_Expression']:
                continue
            
            frame = Image.open(BytesIO(video_info['frames'][0])).resize((self.img_width, self.img_height))
            micro_exp = Image.open(BytesIO(video_info['Micro_Expression'][0])).resize((self.micro_exp_width, self.micro_exp_height))

            X_frames.append(np.array(frame))
            X_micro_exp.append(np.array(micro_exp))
            y.append(video_info['frames_label'][0])

        # Convert lists to numpy arrays
        X_frames = np.array(X_frames, dtype=np.float32)
        X_micro_exp = np.array(X_micro_exp, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)  # reshape y to (batch_size, 1)

        return ((X_frames, X_micro_exp), y)

    def on_epoch_end(self):
        # Shuffle data after each epoch to mix original and manipulated videos
        np.random.shuffle(self.original_videos)
        np.random.shuffle(self.manipulated_videos)

    def get_batch_info(self, index):
        # Utility to fetch and count original and manipulated videos in a batch
        original_batch = self.original_videos[index * self.batch_size // 2:(index + 1) * self.batch_size // 2]
        manipulated_batch = self.manipulated_videos[index * self.batch_size // 2:(index + 1) * self.batch_size // 2]
        
        print(f"Batch {index + 1}:")
        print("  Original videos:", len(original_batch))
        print("  Manipulated videos:", len(manipulated_batch))
        print("  Total videos in batch:", len(original_batch) + len(manipulated_batch))
    def get_num_batches(self):
        return int(np.ceil((self.original_videos + self.manipulated_videos).shape[0] / self.batch_size))
    


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from PIL import Image
import random

class VideoDataGenerator3(Sequence):
    def __init__(self, data_dir, batch_size=16, img_height=224, img_width=224, micro_exp_height=64, micro_exp_width=64):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.micro_exp_height = micro_exp_height
        self.micro_exp_width = micro_exp_width
        
        self.original_dir = os.path.join(data_dir, 'original')
        self.manipulated_dir = os.path.join(data_dir, 'manipulated')
        
        self.original_video_dirs = sorted(os.listdir(os.path.join(self.original_dir, 'facial')))
        self.manipulated_video_dirs = sorted(os.listdir(os.path.join(self.manipulated_dir, 'facial')))
        
        # Combine original and manipulated videos
        self.video_names = self.original_video_dirs + self.manipulated_video_dirs
        self.indexes = np.arange(len(self.video_names))

    def __len__(self):
        # Returns the number of batches per epoch
        return int(np.ceil(len(self.video_names) / self.batch_size))

    def __getitem__(self, index):
        # Get the indexes of the current batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Ensure each batch contains both original and manipulated data
        original_batch_size = self.batch_size // 2
        manipulated_batch_size = self.batch_size - original_batch_size
        
        original_batch = random.sample(self.original_video_dirs, original_batch_size)
        manipulated_batch = random.sample(self.manipulated_video_dirs, manipulated_batch_size)
        
        batch_video_names = original_batch + manipulated_batch
        random.shuffle(batch_video_names)  # Shuffle within the batch to mix original and manipulated data
        
        X_frames = []
        X_micro_exp = []
        y = []

        for video_name in batch_video_names:
            # Determine if it's original or manipulated video
            is_original = video_name in self.original_video_dirs
            video_type = 'original' if is_original else 'manipulated'
            
            # Get the facial and micro expression directories for the video
            facial_dir = os.path.join(self.data_dir, video_type, 'facial', video_name)
            micro_exp_dir = os.path.join(self.data_dir, video_type, 'micro_expression', video_name)

            # List all frames for facial and micro expressions
            facial_frames = sorted(os.listdir(facial_dir))
            micro_exp_frames = sorted(os.listdir(micro_exp_dir))
            
            # Process all frames of the video
            frames = [np.array(Image.open(os.path.join(facial_dir, frame)).resize((self.img_width, self.img_height))) for frame in facial_frames]
            micro_expressions = [np.array(Image.open(os.path.join(micro_exp_dir, frame)).resize((self.micro_exp_width, self.micro_exp_height))) for frame in micro_exp_frames]

            # Normalize the frames and micro expressions by dividing by 255.0
            frames = np.array(frames) / 255.0
            micro_expressions = np.array(micro_expressions) / 255.0
            
            # Append the batch data
            X_frames.append(frames)
            X_micro_exp.append(micro_expressions)
            y.append(1.0 if is_original else 0.0)  # 1.0 for original, 0.0 for manipulated

        # Convert to numpy arrays (ensure correct dtype)
        X_frames = np.array(X_frames, dtype=np.float32)
        X_micro_exp = np.array(X_micro_exp, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # Ensure y is shaped correctly for multi-label classification
        y = y.reshape(-1, 1)  # reshape y to (batch_size * total_frames_in_videos, 1)
        
        return (X_frames, X_micro_exp), y

    def on_epoch_end(self):
        # Shuffle the indexes after each epoch
        np.random.shuffle(self.indexes)
