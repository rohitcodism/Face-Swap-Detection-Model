import tensorflow as tf
from tensorflow.keras.utils import Sequence
from PIL import Image
import numpy as np
from io import BytesIO

class VideoDataGenerator(Sequence):
    def __init__(
        self, data, batch_size=32, sequence_length=30, 
        img_height=224, img_width=224, micro_exp_height=64, micro_exp_width=64, 
        shuffle=True, augment=False
    ):
        self.data = data
        self.batch_size = batch_size
        self.sequence_length = sequence_length  # Number of frames per sequence
        self.img_height = img_height
        self.img_width = img_width
        self.micro_exp_height = micro_exp_height
        self.micro_exp_width = micro_exp_width
        self.shuffle = shuffle
        self.augment = augment
        self.video_names = list(data.keys())
        self.indexes = np.arange(len(self.video_names))
        self.on_epoch_end()

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
            
            frames_seq = []
            micro_exp_seq = []
            
            # Ensure the video has enough frames
            if len(video_info['frames']) < self.sequence_length or len(video_info['Micro_Expression']) < self.sequence_length:
                continue  # Skip or handle padding if necessary
            
            # Load the sequence of frames and micro-expressions
            for i in range(self.sequence_length):
                frame = Image.open(BytesIO(video_info['frames'][i])).convert('RGB').resize((self.img_width, self.img_height))
                micro_exp = Image.open(BytesIO(video_info['Micro_Expression'][i])).convert('RGB').resize((self.micro_exp_width, self.micro_exp_height))
                
                frame_np = np.array(frame) / 255.0  # Normalize frame
                micro_exp_np = np.array(micro_exp) / 255.0  # Normalize micro-expression
                
                frames_seq.append(frame_np)
                micro_exp_seq.append(micro_exp_np)

            X_frames.append(frames_seq)
            X_micro_exp.append(micro_exp_seq)
            y.append(video_info['frame_label'][0])  # Assuming same label for entire sequence
        
        # Convert lists to numpy arrays and ensure correct dtype
        X_frames = np.array(X_frames, dtype=np.float32)
        X_micro_exp = np.array(X_micro_exp, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)  # Adjust labels shape
        
        return ((X_frames, X_micro_exp), y)

    # def on_epoch_end(self):
    #     if self.shuffle:
    #         np.random.shuffle(self.indexes)
