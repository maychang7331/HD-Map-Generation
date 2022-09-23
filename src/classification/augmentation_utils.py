import numpy as np
# import tensorflow as tf
# from keras.preprocessing.image import transform_matrix_offset_center
from keras.preprocessing.image import ImageDataGenerator
# https://zhuanlan.zhihu.com/p/29513760

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix



def rotate(x, theta, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = ImageDataGenerator.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

rotate_limit=(-30, 30)
theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1]) #逆时针旋转角度
#rotate_limit= 30 #自定义旋转角度
#theta = np.pi /180 *rotate_limit #将其转换为PI
# img_rot = rotate(img, theta)