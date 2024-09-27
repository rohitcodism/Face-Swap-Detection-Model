# Load the data from the pickle file
import pickle
from datamaker import VideoDataGenerator
import tensorflow as tf
from deliver import deliver_model
from pipeline import build_full_model
import matplotlib.pyplot as plt


with open('Models/data/video_data_large.pkl', 'rb') as f:
    pickled_data = pickle.load(f)


# Inspect the data
print(type(pickled_data))  # Check the type of the data
print(pickled_data)        # Check the contents of the data


# from sklearn.model_selection import train_test_split

# # Convert your video_data dictionary to a list of items for easier splitting
# data_items = list(pickled_data.items())
# video_names, labels = zip(*[(video_name, video_info['frame_label'][0]) for video_name, video_info in pickled_data.items()])

# # Split the data
# train_names, temp_names, train_labels, temp_labels = train_test_split(video_names, labels, test_size=0.2, random_state=42)
# val_names, test_names, val_labels, test_labels = train_test_split(temp_names, temp_labels, test_size=0.5, random_state=42)

# # Prepare dictionaries for each split
# train_data = {name: pickled_data[name] for name in train_names}
# val_data = {name: pickled_data[name] for name in val_names}
# test_data = {name: pickled_data[name] for name in test_names}

# # Define the output signature for the generator
# output_signature = (
#     (
#         tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
#         tf.TensorSpec(shape=(None, 64, 64, 3), dtype=tf.float32)
#     ),
#     tf.TensorSpec(shape=(None,1), dtype=tf.float32)
# )

# train_generator = tf.data.Dataset.from_generator(
#     lambda: VideoDataGenerator(train_data),
#     output_signature=output_signature
# )

# val_generator = tf.data.Dataset.from_generator(
#     lambda: VideoDataGenerator(val_data),
#     output_signature=output_signature
# )

# test_generator = tf.data.Dataset.from_generator(
#     lambda: VideoDataGenerator(test_data),
#     output_signature=output_signature
# )

# tf.keras.backend.clear_session()

# # build pipeline
# model_test_1 = build_full_model()

# optimizer = tf.keras.optimizers.Adam(clipvalue=1.0)

# # compile the model
# model_test_1.compile(
#     optimizer=optimizer,
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )
# model_test_1.summary()

# #train the model

# history = model_test_1.fit(
#         train_generator,
#         epochs=35,
#         validation_data=val_generator,
#         verbose=2
#     )


# # Evaluate the model on the test data
# test_loss, test_accuracy = model_test_1.evaluate(test_generator)
# print(f"Test Loss: {test_loss}")
# print(f"Test Accuracy: {test_accuracy}")

# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()