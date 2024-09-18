import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
from io import BytesIO

class VideoDataGenerator(Sequence):
    def __init__(self, data, batch_size=32, sequence_length=30, img_height=224, img_width=224, micro_exp_height=64, micro_exp_width=64):
        self.data = data
        self.batch_size = batch_size
        self.sequence_length = sequence_length
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
            
            # Load frames and micro expressions as images
            frames = [Image.open(BytesIO(img_bytes)).resize((self.img_width, self.img_height)) for img_bytes in video_info['frames']]
            micro_exp = [Image.open(BytesIO(img_bytes)).resize((self.micro_exp_width, self.micro_exp_height)) for img_bytes in video_info['Micro_Expression']]
            
            # Convert to numpy arrays and ensure consistent shape
            frames_np = np.array([np.array(img) for img in frames])
            micro_exp_np = np.array([np.array(img) for img in micro_exp])
            
            # Ensure consistent sequence length
            if len(frames_np) < self.sequence_length:
                frames_np = np.pad(frames_np, ((0, self.sequence_length - len(frames_np)), (0, 0), (0, 0), (0, 0)), mode='constant')
            else:
                frames_np = frames_np[:self.sequence_length]
                
            if len(micro_exp_np) < self.sequence_length:
                micro_exp_np = np.pad(micro_exp_np, ((0, self.sequence_length - len(micro_exp_np)), (0, 0), (0, 0), (0, 0)), mode='constant')
            else:
                micro_exp_np = micro_exp_np[:self.sequence_length]
            
            X_frames.append(frames_np)
            X_micro_exp.append(micro_exp_np)
            y.append(video_info['frame_label'][0])  # Adjust if needed for multiple labels
        
        # Convert to numpy arrays (ensure correct dtype)
        X_frames = np.array(X_frames, dtype=np.float32)
        X_micro_exp = np.array(X_micro_exp, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        return ((X_frames, X_micro_exp), y)

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)


# import pickle

# # Load the data from the pickle file
# with open('Models/video_data_2.pkl', 'rb') as f:
#     pickled_data = pickle.load(f)

# from sklearn.model_selection import train_test_split

# # Convert your video_data dictionary to a list of items for easier splitting
# data_items = list(pickled_data.items())
# video_names, labels = zip(*[(video_name, video_info['frame_label'][0]) for video_name, video_info in pickled_data.items()])

# # Split the data
# train_names, temp_names, train_labels, temp_labels = train_test_split(video_names, labels, test_size=0.3, random_state=42)
# val_names, test_names, val_labels, test_labels = train_test_split(temp_names, temp_labels, test_size=0.5, random_state=42)

# # Prepare dictionaries for each split
# train_data = {name: pickled_data[name] for name in train_names}

# # Define the output signature for the generator
# output_signature = (
#     (
#         tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
#         tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32)
#     ),
#     tf.TensorSpec(shape=(None,), dtype=tf.float32)
# )

# train_generator = tf.data.Dataset.from_generator(
#     lambda: VideoDataGenerator(train_data),
#     output_signature=output_signature
# )

