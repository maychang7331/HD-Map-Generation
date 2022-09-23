import os
import sys
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import outdoor3d_util
import data_prep_util

# Constants
data_dir = os.path.join(BASE_DIR, 'data')
npy_data_dir = os.path.join(data_dir, 'shalun_npy/ori')
NUM_POINT = 4096
H5_BATCH_SIZE = 1000
data_dim = [NUM_POINT, 9]       ###########################################################
label_dim = [NUM_POINT]
data_dtype = 'float64'
label_dtype = 'uint8'

# Set paths
filelist = os.path.join(BASE_DIR, 'meta/shalun_npy_ori.txt')
data_label_files = [os.path.join(npy_data_dir, line.rstrip()) for line in open(filelist)]
output_dir = os.path.join(data_dir, 'shalun_h5/'+filelist.split('.')[0].split('_')[-1])
if not os.path.exists(output_dir):
    os.makedirs(output_dir)     # mkdir with multiple depth
output_filename_prefix = os.path.join(output_dir, 'all_data_original_z')
output_block_filelist = os.path.join(output_dir, 'block_filelist.txt')
fout_block = open(output_block_filelist, 'w')

# --------------------------------------
# ----- BATCH WRITE TO HDF5 -----
# --------------------------------------
batch_data_dim = [H5_BATCH_SIZE] + data_dim     # [1000, 4096, 9] 
batch_label_dim = [H5_BATCH_SIZE] + label_dim   # [1000, 4096]
h5_batch_data = np.zeros(batch_data_dim, dtype = np.float64)
h5_batch_label = np.zeros(batch_label_dim, dtype = np.uint8)
buffer_size = 0  # state: record how many samples are currently in buffer
h5_index = 0 # state: the next h5 file to save

def insert_batch(data, label, last_batch = False):
    global h5_batch_data, h5_batch_label
    global buffer_size, h5_index
    data_size = data.shape[0]
    # If there is enough space, just insert
    if buffer_size + data_size <= h5_batch_data.shape[0]:
        h5_batch_data[buffer_size:buffer_size+data_size, ...] = data
        h5_batch_label[buffer_size:buffer_size+data_size] = label
        buffer_size += data_size
    else: # not enough space
        capacity = h5_batch_data.shape[0] - buffer_size     # 還要多少筆block才會補滿該batch
        assert(capacity>=0)
        if capacity > 0:
           h5_batch_data[buffer_size:buffer_size+capacity, ...] = data[0:capacity, ...] 
           h5_batch_label[buffer_size:buffer_size+capacity, ...] = label[0:capacity, ...] 
        # Save batch data and label to h5 file, reset buffer_size
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        data_prep_util.save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype) 
        print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))
        h5_index += 1
        buffer_size = 0
        # recursive call 把還沒寫的block寫到新的h5檔
        insert_batch(data[capacity:, ...], label[capacity:, ...], last_batch)
    if last_batch and buffer_size > 0:
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        data_prep_util.save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
        h5_index += 1
        buffer_size = 0
    return


sample_cnt = 0
for i, data_label_filename in enumerate(data_label_files):
    print(data_label_filename)
    data, label = outdoor3d_util.district2blocks_wapper_normalized(data_label_filename, NUM_POINT, block_size=2.5, stride=1.0,
                                                 random_sample=False, sample_num=None)
    if len(data)!=0:
        print('{0}, {1}'.format(data.shape, label.shape))
        for _ in range(data.shape[0]):      # 把等於block數之filename寫入txt
            fout_block.write(os.path.basename(data_label_filename)[0:-4]+'\n')
            fout_block.flush()              # 更新txt(按F5)
    

        sample_cnt += data.shape[0]         # 加總block數
        insert_batch(data, label, i == len(data_label_files)-1)
    
    elif i == len(data_label_files)-1:
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        data_prep_util.save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
    
    else:
        continue
    
fout_block.close()
print("Total samples:{0}".format(sample_cnt))   # 即總block數

# 輸出 output_dir 中所有.h5的路徑
output_h5_filelist = os.path.join(output_dir, 'all_h5files.txt')
fout_h5files = open(output_h5_filelist, 'w')
for file in sorted(os.listdir(output_dir)):
    if file.endswith(".h5"):
        fout_h5files.write(output_dir + '/' + file + '\n')
fout_h5files.close()