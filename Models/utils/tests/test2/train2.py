import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
from io import BytesIO
import pickle
from Models.utils.tests.test2.datamaker2 import VideoDataGenerator  # Ensure this points to the updated generator
from Models.utils.tests.test2.pipeline2 import build_full_model  # Ensure this points to the revised model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Load the data from the pickle file
with open('Models/data/video_data_large_2.pkl', 'rb') as f:
    pickled_data = pickle.load(f)

# Extract video names and labels
video_names, labels = zip(*[
    (video_name, video_info['frame_label'][0]) 
    for video_name, video_info in pickled_data.items()
])

# Check label consistency (optional but recommended)
inconsistent_labels = []
for video_name, video_info in pickled_data.items():
    frame_labels = video_info['frame_label']
    if len(set(frame_labels)) > 1:
        inconsistent_labels.append(video_name)

if inconsistent_labels:
    print(f"Videos with inconsistent labels: {inconsistent_labels}")
else:
    print("All videos have consistent labels.")

# Split the data into training, validation, and test sets
train_names, temp_names, train_labels, temp_labels = train_test_split(
    video_names, labels, test_size=0.2, random_state=42, stratify=labels
)
val_names, test_names, val_labels, test_labels = train_test_split(
    temp_names, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

# Prepare dictionaries for each split
train_data = {name: pickled_data[name] for name in train_names}
val_data = {name: pickled_data[name] for name in val_names}
test_data = {name: pickled_data[name] for name in test_names}

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")

# Define the output signature for the generator
output_signature = (
    (
        tf.TensorSpec(shape=(None, 30, 224, 224, 3), dtype=tf.float32),  # 30 frames per sequence
        tf.TensorSpec(shape=(None, 30, 64, 64, 3), dtype=tf.float32)    # 30 micro-expressions per sequence
    ),
    tf.TensorSpec(shape=(None,1), dtype=tf.float32)  # Labels (binary classification)
)

# Create tf.data.Dataset generators
train_generator = tf.data.Dataset.from_generator(
    lambda: VideoDataGenerator(
        train_data, 
        batch_size=32, 
        sequence_length=30, 
        img_height=224, 
        img_width=224, 
        micro_exp_height=64, 
        micro_exp_width=64, 
        shuffle=True, 
        augment=False  # Set to True if you have augmentation implemented
    ),
    output_signature=output_signature
).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

val_generator = tf.data.Dataset.from_generator(
    lambda: VideoDataGenerator(
        val_data, 
        batch_size=32, 
        sequence_length=30, 
        img_height=224, 
        img_width=224, 
        micro_exp_height=64, 
        micro_exp_width=64, 
        shuffle=False, 
        augment=False
    ),
    output_signature=output_signature
).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

test_generator = tf.data.Dataset.from_generator(
    lambda: VideoDataGenerator(
        test_data, 
        batch_size=32, 
        sequence_length=30, 
        img_height=224, 
        img_width=224, 
        micro_exp_height=64, 
        micro_exp_width=64, 
        shuffle=False, 
        augment=False
    ),
    output_signature=output_signature
).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Build the model
model_test_1 = build_full_model()

# Compile the model
optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)  # Clip gradients by norm

model_test_1.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_test_1.summary()

# # Define callbacks
# early_stopping = EarlyStopping(
#     monitor='val_loss',
#     patience=10,
#     restore_best_weights=True,
#     verbose=1
# )

# lr_scheduler = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.5,
#     patience=5,
#     verbose=1,
#     min_lr=1e-6
# )

# # Train the model with callbacks
# history = model_test_1.fit(
#     train_generator,
#     epochs=50,
#     validation_data=val_generator,
#     callbacks=[early_stopping, lr_scheduler],
#     # Remove batch_size since tf.data.Dataset handles batching
# )

# # Evaluate the model on the test data
# test_loss, test_accuracy = model_test_1.evaluate(test_generator)
# print(f"Test Loss: {test_loss}")
# print(f"Test Accuracy: {test_accuracy}")

# # Plot training and validation loss and accuracy
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Loss Curves')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Accuracy Curves')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.show()
