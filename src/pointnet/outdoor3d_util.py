import numpy as np
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

g_classes = [x.rstrip() for x in open(os.path.join(BASE_DIR, 'meta/class_names.txt'))]
g_class2label = {cls: i for i,cls in enumerate(g_classes)}
g_class2color = {'other':           [0, 0, 255],        # 0000FF
                 'traffic island':  [255, 136, 235],    # FF8AEB
                 'sign':            [255, 255, 0],      # FFFF00
                 'signal':          [46, 196, 182],     # 2EC2B3
                 'pole':            [231, 29, 54]}      # E71D34
g_label2color = {g_classes.index(cls): g_class2color[cls] for cls in g_classes}

# -----------------------------------------------------------------------------
# CONVERT ORIGINAL DATA(.txt) TO DATA_LABEL FILES(.npy)
# -----------------------------------------------------------------------------

def modifyclassID(id_array):
    for index, classid in enumerate(id_array):
        if classid == 1:        # other
            id_array[index] = 0
        elif classid == 23:     # traffic island
            id_array[index] = 1         
        elif classid == 24:     # sign
            id_array[index] = 2
        elif classid == 25:     # signal
            id_array[index] = 3
        elif classid == 26:     # pole
            id_array[index] = 4
        else:
            print('undefined class id !')

def collect_point_label(anno_path, out_filename, file_format='txt'):
    """
    Args:
        anno_path: path to annotations. e.g. /home/chihyu/Desktop/thesis/data/shalun_txt/9/pt_9.txt
        out_filename: filename of collected points and labels (each line is XYZIRGBL)
        file_format: txt only
    Note:
        the points are shifted before save, the most negative point is now at origin.
    """
    
    if anno_path.endswith('.txt'):
        data_label = np.loadtxt(anno_path)          # XYZIRGBL
        modifyclassID(data_label[:, -1])
        xy_min = np.amin(data_label, axis=0)[0:2]   # 獲取每一行(axis=0)的最小值, 即x, y之最小值
        # xy_min = [176900, 2535700]                # target only
        shiftxy = data_label[:, 0:2] - xy_min
        data_label = np.hstack((data_label[:, 0:2], shiftxy, data_label[:, 2:]))   # X Y X' Y' Z I R G B L
        if file_format == 'txt':
            fout = open(out_filename, 'w')
            for i in range(data_label.shape[0]):
                fout.write('%f %f %f %f %f %f %d %d %d %d \n' % \
                            (data_label[i,0], data_label[i,1], data_label[i,2],
                             data_label[i,3], data_label[i,4], data_label[i,5],
                             data_label[i,6], data_label[i,7], data_label[i,8], data_label[i,9]))
            fout.close()
        
        elif file_format=='numpy':
            np.save(out_filename, data_label)       # dimension: X Y X' Y' Z I R G B L ( NX10 )

        else:
            print('ERROR!! Unknown file format: %s, please use txt or numpy.' % (file_format))
            exit()
    else:
        print('ERROR!! Unknown input file format, please use txt')
        exit()

def sample_data(data, num_sample):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    if (N == num_sample):
        return data, range(N)
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)    # 從0~N, 隨機選出 num_sample(4096) 個
        return data[sample, ...], sample
    else:
        sample = np.random.choice(N, num_sample-N)
        dup_data = data[sample, ...]
        return np.concatenate([data, dup_data], 0), list(range(N)) + sample.tolist()

def sample_data_label(data, label, num_sample):
    new_data, sample_indices = sample_data(data, num_sample)
    new_label = label[sample_indices]
    return new_data, new_label  # (4096, 9) (4096)

def district2blocks(data, label, num_point, block_size=1.0, stride=1.0, 
               random_sample=False, sample_num=None, sample_aug=1):
    """ Prepare block training data.
        Note that X, Y are shifted (origin set at each block center)
    Args:
        data: N x 7 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (origin set at min) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-4
        num_point: int, how many points to sample in each block (4096)
        block_size: float, physical size of the block in meters
        stride: float, stride for block sweeping
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: total_block x num_point(4096) x 7 np array of X'Y'ZRGBI, RGB is in [0,1]
        block_labels: total_block x num_point(4096) x 1 np array of uint8 labels
    """
    assert(stride<=block_size)          # 斷言，不然會有格子被跳過

    limit = np.amax(data, 0)[2:5]       # X' Y' Z 的最大值

    # Get the corner location for our sampling blocks    
    xbeg_list = []
    ybeg_list = []
    if not random_sample:
        num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
        num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
        for i in range(num_block_x):
            for j in range(num_block_y):
                xbeg_list.append(i*stride)
                ybeg_list.append(j*stride)
    else:
        num_block_x = int(np.ceil(limit[0] / block_size))
        num_block_y = int(np.ceil(limit[1] / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y * sample_aug
        for _ in range(sample_num):
            xbeg = np.random.uniform(-block_size, limit[0]) 
            ybeg = np.random.uniform(-block_size, limit[1]) 
            xbeg_list.append(xbeg)
            ybeg_list.append(ybeg)

    # Collect blocks
    block_data_list = []
    block_label_list = []
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
        # block_data[:, 4] -= min(block_data[:, 4])     # 平移z座標移至該block最低點
        # print((xbeg + (xbeg+block_size))/2, (ybeg + (ybeg+block_size))/2)
        # print(block_data[0:3, 0:4])

        # 每個block採樣4096個點
        block_data_sampled, block_label_sampled = sample_data_label(block_data, block_label, num_point)
        
        
        # 切換是否只輸入不全為other的blocks
        ''' <所有blocks>
        '''
        block_data_list.append(np.expand_dims(block_data_sampled, 0))
        block_label_list.append(np.expand_dims(block_label_sampled, 0))
        
        ''' <不全為other的blocks, 只放有target的blocks> 
        
        if (block_label_sampled == 0).sum() != num_point:
            block_data_list.append(np.expand_dims(block_data_sampled, 0))
            block_label_list.append(np.expand_dims(block_label_sampled, 0))
        else:
            continue
        '''

    if len(block_data_list) != 0:    # 不是 empty 才 concatenate
        return np.concatenate(block_data_list, 0), \
                np.concatenate(block_label_list, 0)
    else:
        return block_data_list, block_label_list    # return 兩個空的list [], []

def district2blocks_plus_normalized(data_label, num_point, block_size, stride, random_sample, 
                                     sample_num, sample_aug):
    """ 
    """
    data = data_label[:, 0:9]       # X Y X' Y' Z I R G B L (N X 10)
    data[:, 5] /= 65535.0           # 正規化intensity
    data[:, 6:9] /= 65535.0         # 正規化顏色到[0,1]
    label = data_label[:, -1].astype(np.uint8)
    # max_dis_x = max(data[:,0])
    # max_dis_y = max(data[:,1])
    # max_dis_z = max(data[:,2])

    data_batch, label_batch = district2blocks(data, label, num_point, block_size, stride, 
                                         random_sample, sample_num, sample_aug)
    
    # new_data_batch = np.zeros((data_batch.shape[0], num_point, 10))
    # for b in range(data_batch.shape[0]):
    #     new_data_batch[b, :, 7] = data_batch[b, :, 0]/max_dis_x
    #     new_data_batch[b, :, 8] = data_batch[b, :, 1]/max_dis_y
    #     new_data_batch[b, :, 9] = data_batch[b, :, 2]/max_dis_z
    #     minx = min(data_batch[b, :, 0])
    #     miny = min(data_batch[b, :, 1])
    #     data_batch[b, :, 0] -= (minx+block_size/2)  # 把每個block的xy平面座標的中心移至
    #     data_batch[b, :, 1] -= (miny+block_size/2)
    # new_data_batch[:, :, 0:7] = data_batch
    return data_batch, label_batch


def district2blocks_wapper_normalized(data_label_filename, num_point, block_size=1.0, stride=1.0,
                                       random_sample=False, sample_num=None, sample_aug=1):
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
    return district2blocks_plus_normalized(data_label, num_point, block_size, stride, random_sample, 
                                            sample_num, sample_aug)