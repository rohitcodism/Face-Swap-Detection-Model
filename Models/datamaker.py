from io import BytesIO
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence

class VideoDataGenerator(Sequence):
    def __init__(self, data, batch_size=32, sequence_length=30):
        self.data = data
        self.batch_size = batch_size
        self.sequence_length = sequence_length
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
            frames = np.array([Image.open(BytesIO(img_bytes)) for img_bytes in video_info['frames']])
            micro_exp = np.array([Image.open(BytesIO(img_bytes)) for img_bytes in video_info['Micro_Expression']])
            frame_labels = video_info['frame_label']
            micro_exp_labels = video_info['Micro_Expression_label']
            
            # Ensure frames and micro expressions are sequences of the required length
            if len(frames) < self.sequence_length:
                frames = np.pad(frames, ((0, self.sequence_length - len(frames)), (0, 0), (0, 0)), mode='constant')
            else:
                frames = frames[:self.sequence_length]
                
            if len(micro_exp) < self.sequence_length:
                micro_exp = np.pad(micro_exp, ((0, self.sequence_length - len(micro_exp)), (0, 0), (0, 0)), mode='constant')
            else:
                micro_exp = micro_exp[:self.sequence_length]
            
            X_frames.append(frames)
            X_micro_exp.append(micro_exp)
            y.append(frame_labels[0])  # Adjust if needed for multiple labels
        
        return [np.array(X_frames), np.array(X_micro_exp)], np.array(y)

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)
