from function import load_4d_array_from_hdd
import pickle

def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

video_data = load_4d_array_from_hdd("/Users/piyush/Desktop/Data")

# After processing each batch, save it to disk
save_data(video_data, f"video_data_large.pkl")


print("Okay!! Done")