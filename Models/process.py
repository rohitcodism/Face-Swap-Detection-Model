import numpy as np

def preprocess_data(video_data, seq_length):
    X_facial = []
    X_micro_exp = []
    y = []

    for video, data in video_data.items():
        frames = np.array([np.array(frame) for frame in data['frames']])
        frame_labels = np.array(data['frame_label'])
        micro_exp = np.array([np.array(micro) for micro in data['Micro_Expression']])
        micro_exp_labels = np.array(data['Micro_Expression_label'])

        # Ensure we have enough frames for the sequences
        if len(frames) >= seq_length:
            # Create temporal sequences for facial frames
            facial_sequences = [frames[i:i + seq_length] for i in range(len(frames) - seq_length + 1)]
            facial_labels = [frame_labels[i + seq_length - 1] for i in range(len(frames) - seq_length + 1)]

            # Create temporal sequences for micro-expressions
            micro_exp_sequences = [micro_exp[i:i + seq_length] for i in range(len(micro_exp) - seq_length + 1)]
            micro_exp_labels = [micro_exp_labels[i + seq_length - 1] for i in range(len(micro_exp) - seq_length + 1)]

            # Append sequences and labels to dataset
            X_facial.extend(facial_sequences)
            X_micro_exp.extend(micro_exp_sequences)
            y.extend(facial_labels)  # Assuming facial frame labels are used for the final classification

    return np.array(X_facial), np.array(X_micro_exp), np.array(y)

seq_length = 30  # Example sequence length

