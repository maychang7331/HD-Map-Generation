import os
import sys
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import h5py

# Write numpy array data and label to h5_filename
def save_h5(h5_filename, data, label, data_dtype='float64', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename, 'w')

    # 使用gzip壓縮，壓縮等級1~9 [default:4] (壓縮等級 1 表示壓縮速度最快, 但是壓縮比最差)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=9,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=9,
            dtype=label_dtype)
    h5_fout.close()