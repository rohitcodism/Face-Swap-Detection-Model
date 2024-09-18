# Load the data from the pickle file
import pickle
from datamaker import VideoDataGenerator
import tensorflow as tf
from deliver import deliver_model


with open('Models/video_data_2.pkl', 'rb') as f:
    pickled_data = pickle.load(f)

from sklearn.model_selection import train_test_split

# Convert your video_data dictionary to a list of items for easier splitting
data_items = list(pickled_data.items())
video_names, labels = zip(*[(video_name, video_info['frame_label'][0]) for video_name, video_info in pickled_data.items()])

# Split the data
train_names, temp_names, train_labels, temp_labels = train_test_split(video_names, labels, test_size=0.3, random_state=42)
val_names, test_names, val_labels, test_labels = train_test_split(temp_names, temp_labels, test_size=0.5, random_state=42)

# Prepare dictionaries for each split
train_data = {name: pickled_data[name] for name in train_names}
val_data = {name: pickled_data[name] for name in val_names}
test_data = {name: pickled_data[name] for name in test_names}

# Define the output signature for the generator
output_signature = (
    (
        tf.TensorSpec(shape=(None, 30, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 30, 64, 64, 3), dtype=tf.float32)
    ),
    tf.TensorSpec(shape=(None,), dtype=tf.float32)
)

train_generator = tf.data.Dataset.from_generator(
    lambda: VideoDataGenerator(train_data),
    output_signature=output_signature
)

val_generator = tf.data.Dataset.from_generator(
    lambda: VideoDataGenerator(val_data),
    output_signature=output_signature
)

test_generator = tf.data.Dataset.from_generator(
    lambda: VideoDataGenerator(test_data),
    output_signature=output_signature
)



model_test_1 = deliver_model()

model_test_1.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)