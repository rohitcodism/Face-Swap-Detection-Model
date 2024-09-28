import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
from io import BytesIO

class VideoDataGenerator(Sequence):
    def __init__(self, data, batch_size=32, sequence_length=30, img_height=224, img_width=224, micro_exp_height=64, micro_exp_width=64):
        self.data = data
        self.batch_size = batch_size
        self.sequence_length = sequence_length  # New parameter for sequence length
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
            
            frames_seq = []
            micro_exp_seq = []
            
            # Load the sequence of frames and micro-expressions
            for i in range(self.sequence_length):
                frame = Image.open(BytesIO(video_info['frames'][i])).resize((self.img_width, self.img_height))
                micro_exp = Image.open(BytesIO(video_info['Micro_Expression'][i])).resize((self.micro_exp_width, self.micro_exp_height))
                
                frame_np = np.array(frame)
                micro_exp_np = np.array(micro_exp)
                
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
    #     np.random.shuffle(self.indexes)