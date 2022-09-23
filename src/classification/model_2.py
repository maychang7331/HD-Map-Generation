import tensorflow as tf
import tensorflow.contrib.layers as layers


def Inception(inputs, conv11_size, conv33_11_size, conv33_size, conv55_11_size, conv55_size, pool11_size):
    conv11 = layers.conv2d(inputs, conv11_size, [1, 1])
    
    conv33_reduce = layers.conv2d(inputs, conv33_11_size, [1, 1])   
    conv33 = layers.conv2d(conv33_reduce, conv33_size, [3, 3])      
    
    conv55_reduce = layers.conv2d(inputs, conv55_11_size, [1, 1])   
    conv55 = layers.conv2d(conv55_reduce, conv55_size, [5, 5])      
    
    pool_proj = layers.max_pool2d(inputs, [3, 3], stride = 1, padding='SAME')
    pool11 = layers.conv2d(pool_proj, pool11_size, [1, 1])
    
    return tf.concat([conv11, conv33, conv55, pool11], 3)

def GoogLeNet(inputs, dropout_keep_prob, num_of_classes): # inputs size:32x32x3
    conv1 = layers.conv2d(inputs, 64, [3, 3], stride = 2) # 16x16x64        
    
    inception_2a = Inception(conv1, 64, 96, 128, 16, 32, 32) # 16x16x256
    inception_2b = Inception(inception_2a, 128, 128, 192, 32, 96, 64) # 16x16x480
    pool2 = layers.max_pool2d(inception_2b, [3, 3]) # 7x7x480 ? why
    
    inception_3a = Inception(pool2, 192, 96, 208, 16, 48, 64) # 7x7x512
    inception_3b = Inception(inception_3a, 160, 112, 224, 24, 64, 64) # 7x7x512
    pool3 = layers.max_pool2d(inception_3b, [3, 3]) # 3x3x512
    
    inception_4a = Inception(pool3, 256, 160, 320, 32, 128, 128) # 3x3x832
    inception_4b = Inception(inception_4a, 384, 192, 384, 48, 128, 128) # 3x3x1024
    pool4 = layers.avg_pool2d(inception_4b, [3, 3], stride = 1)

    reshape = tf.reshape(pool4, [-1, 1024])
    dropout = layers.dropout(reshape, dropout_keep_prob)
    logits = layers.fully_connected(dropout, num_of_classes, activation_fn=None)
    
    return logits