from function import load_4d_array_from_hdd, load_and_save_data_incrementally
import pickle

def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

load_and_save_data_incrementally("/Users/piyush/Desktop/Data")


print("Okay!! Done")