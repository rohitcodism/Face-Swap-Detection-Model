import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
from io import BytesIO

class VideoDataGenerator(Sequence):
    def __init__(self, data, batch_size=32, img_height=224, img_width=224, micro_exp_height=64, micro_exp_width=64):
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
            
            # Load the first frame and micro expression as images
            frame = Image.open(BytesIO(video_info['frames'][0])).resize((self.img_width, self.img_height))
            micro_exp = Image.open(BytesIO(video_info['Micro_Expression'][0])).resize((self.micro_exp_width, self.micro_exp_height))
            
            # Convert to numpy arrays
            frame_np = np.array(frame)
            micro_exp_np = np.array(micro_exp)
            
            X_frames.append(frame_np)
            X_micro_exp.append(micro_exp_np)
            y.append(video_info['frame_label'][0])  # Adjust if needed for multiple labels
        
        # Convert to numpy arrays (ensure correct dtype)
        X_frames = np.array(X_frames, dtype=np.float32)
        X_micro_exp = np.array(X_micro_exp, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        y = y.reshape(-1, 1) # reshape y to (batch_size, 1)
        
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

