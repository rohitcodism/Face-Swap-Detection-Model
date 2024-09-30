import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
from io import BytesIO

class VideoDataGenerator(Sequence):
    def _init_(self, data, batch_size=32, img_height=224, img_width=224, micro_exp_height=64, micro_exp_width=64):
        self.data = data
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.micro_exp_height = micro_exp_height
        self.micro_exp_width = micro_exp_width
        self.video_names = list(data.keys())
        self.indexes = np.arange(len(self.video_names))

    def _len_(self):
        return int(np.ceil(len(self.video_names) / self.batch_size))

    def _getitem_(self, index):
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
            
            # Normalize the frames and micro expressions by dividing by 255.0
            # frame_np = frame_np / 255.0
            # micro_exp_np = micro_exp_np / 255.0
            
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