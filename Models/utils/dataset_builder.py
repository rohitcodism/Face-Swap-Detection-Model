from function import load_and_save_data_incrementally, load_and_save_data
import pickle
import h5py
import numpy as np
import pandas as pd





# video_dataframe = pd.DataFrame(video_data_test1)


# def assign_video_label(frame_labels, micro_expression_labels):
#     # Any frame or micro-expression being manipulated sets video as manipulated (1)
#     if np.any(np.array(frame_labels) == 1) or np.any(np.array(micro_expression_labels) == 1):
#         return 1
#     else:
#         return 0
    
# video_data_test1 = load_4d_array_from_hdd("/Users/piyush/Desktop/Data")
    
# video_dataframe = pd.DataFrame(video_data_test1)


# def save_data(data, filename):
#     with open(filename, 'wb') as f:
#         pickle.dump(data, f)

load_and_save_data("G:/Preprocessed Data")

# save_data(video_data, "/Volumes/E/video_data_large_2.pkl")


print("Okay!! Done")