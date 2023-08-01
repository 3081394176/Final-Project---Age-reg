import pickle
import os
import h5py
import numpy as np
import os

import h5py
import numpy as np


def save_data(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
        print(file_name, "saved")

def load_data(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"{file_name} does not exist.")
        return None

