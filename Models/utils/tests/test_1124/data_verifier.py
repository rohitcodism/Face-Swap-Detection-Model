import pickle

# Load the dataset from the pickle file
with open("D:/Projects/Face-Swap-Detection-Model/video_data_2600_001.pkl", "rb") as f:
    video_data = pickle.load(f)

# Print top-level structure
print("Top-level keys (video folders):", list(video_data.keys())[:5])  # Show first 5 video folders

# Print the structure of one sample entry
sample_video = list(video_data.keys())[0]  # Take the first video folder as an example
print(f"Sample video '{sample_video}' structure:")
print("Frames count:", len(video_data[sample_video]['frames']))
print("Frames labels count:", len(video_data[sample_video]['frames_label']))
print("Micro-expression count:", len(video_data[sample_video]['Micro_Expression']))
print("Micro-expression labels count:", len(video_data[sample_video]['Micro_Expression_label']))


# Verify consistency across all video folders
for video_folder, data in video_data.items():
    assert len(data['frames']) == len(data['frames_label']), f"Mismatch in frames and labels for {video_folder}"
    assert len(data['Micro_Expression']) == len(data['Micro_Expression_label']), f"Mismatch in micro expressions and labels for {video_folder}"
print("All data and label pairs are consistent in length.")


from PIL import Image
import io

# Decode and show a sample frame from the first video folder
sample_frame_bytes = video_data[sample_video]['frames'][0]
sample_image = Image.open(io.BytesIO(sample_frame_bytes))
sample_image.show()

# Decode and show a sample micro expression from the first video folder
sample_micro_expression_bytes = video_data[sample_video]['Micro_Expression'][0]
sample_image = Image.open(io.BytesIO(sample_micro_expression_bytes))
sample_image.show()