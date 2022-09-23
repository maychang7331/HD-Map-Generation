import h5py
import numpy as np

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

# Read numpy array data and label from h5_filename
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """
    print(batch_data.shape)
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    print(batch_data)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        rotation_angle = rotation_angle * np.pi/180
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],                            
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])                      
        
        rotated_data_T = rotation_matrix.dot(np.transpose(batch_data))
        rotated_data = np.transpose(rotated_data_T)
        print(rotated_data)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data