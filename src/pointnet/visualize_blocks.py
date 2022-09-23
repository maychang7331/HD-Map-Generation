import os
import sys
import numpy as np
import outdoor3d_util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

# Constants
data_dir = os.path.join(BASE_DIR, 'data')
npy_data_dir = os.path.join(data_dir, 'shalun_npy/dso8')
num_point = 4096

# Set paths
filelist = os.path.join(BASE_DIR, 'meta/shalun_npy_dso8.txt')
data_label_files = [os.path.join(npy_data_dir, line.rstrip()) for line in open(filelist)]

output_dir = os.path.join(data_dir, 'shalun_visblks/dso8')
if not os.path.exists(output_dir):
    os.makedirs(output_dir) 


def district2blocks_wapper_normalized(data_label_filename, output_district_dir, block_size=1.0, stride=1.0,):
    
    '''
    return
        data  : (block_num, 4096, 9)    X Y X" Y" Z I' R' G' B'
        label : (block_num, 4096)       L
    '''

    if data_label_filename[-3:] == 'npy':
        data_label = np.load(data_label_filename)
    else:
        print('Unknown file type! exiting.')
        exit()
    
    data = data_label[:, 0:9]       # X Y X' Y' Z I R G B L (N X 10)
    data[:, 5] /= 65535.0           # 正規化intensity
    data[:, 6:9] /= 65535.0         # 正規化顏色到[0,1]
    label = data_label[:, -1].astype(np.uint8)


    assert(stride<=block_size)          # 斷言，不然會有格子被跳過
    
    limit = np.amax(data, 0)[2:5]       # X' Y' Z 的最大值

    # Get the corner location for our sampling blocks    
    xbeg_list = []
    ybeg_list = []
    num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
    num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
    for i in range(num_block_x):
        for j in range(num_block_y):
            xbeg_list.append(i*stride)
            ybeg_list.append(j*stride)

    idx = 0
    for idx in range(len(xbeg_list)):       # len(xbeg_list) = xbeg_list * ybeg_list
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]
        xcond = (data[:,2]<=xbeg+block_size) & (data[:,2]>=xbeg)    # 符合座標區間的所有點 True False Matrix
        ycond = (data[:,3]<=ybeg+block_size) & (data[:,3]>=ybeg)    # 符合座標區間的所有點 True False Matrix
        cond = xcond & ycond    # 由比對 xcond 和 ycond而得的 true false矩陣
        if np.sum(cond) == 0:   # discard block if there isn't any point. (原: < 100)
            continue

        block_data = data[cond, :]
        block_label = label[cond]
        # print(block_data[0:3, 0:4])

        # 平移x, y座標至block中心
        block_data[:, 2] -= (xbeg + (xbeg+block_size))/2
        block_data[:, 3] -= (ybeg + (ybeg+block_size))/2



        ## every points in the block
        """
        fout_block_filename = os.path.join(output_district_dir, str(idx))+'.txt'
        fout_block = open(fout_block_filename, 'w')

        for k in range(len(block_label)):
            color = outdoor3d_util.g_label2color[block_label[k]]
            fout_block.write('%f %f %f %f %d %d %d \n' % (block_data[k,0], block_data[k,1], block_data[k,4],
                                                            block_data[k,5], color[0], color[1], color[2]))
            fout_block.flush()
        fout_block.close()
        """

        ## sample only blocks with targets
        """"""
        if np.sum(block_label) != 0:
            fout_block_filename = os.path.join(output_district_dir, str(idx))+'.txt'
            fout_block = open(fout_block_filename, 'w')
            block_data_sampled, block_label_sampled = outdoor3d_util.sample_data_label(block_data, block_label, num_point)
            for k in range(len(block_label_sampled)):
                color = outdoor3d_util.g_label2color[block_label_sampled[k]]
                fout_block.write('%f %f %f %f %d %d %d \n' % (block_data_sampled[k,0], block_data_sampled[k,1], block_data_sampled[k,4],
                                                                block_data_sampled[k,5], color[0], color[1], color[2]))
                fout_block.flush()
            fout_block.close()

# for i, data_label_filename in enumerate(data_label_files):
    # i = 8
data_label_filename = '/home/chihyu/Desktop/thesis_2/pointnet/data/shalun_npy/dso8/shalun_dso8_pt_9.npy'
    
district_filename = data_label_filename.split('/')[-1].split('.')[0]    # shalun_ori_pt_9
output_district_dir = os.path.join(output_dir, district_filename)+ ' _sample4096'
if not os.path.exists(output_district_dir):
    os.makedirs(output_district_dir)

data, label = district2blocks_wapper_normalized(data_label_filename, output_district_dir, block_size=2.5, stride=1)
                                                    