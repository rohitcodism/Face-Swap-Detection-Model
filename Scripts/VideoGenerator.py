from keras.api.utils import Sequence
import pandas as pd

class VideoGenerator(Sequence):
    def __init__(self, video_paths, labels, batch_size, num_frames, frame_size=(224,224)):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.frame_size = frame_size

    def __len__(self):
        return len(self.video_paths)
    def __getitem__(self, idx):
        batch_video_paths = self.video_paths[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_labels = self.labels[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_data = []
        for video_path in batch_video_paths:
            frames = frames