import tensorflow as tf
from tensorflow.contrib.layers import flatten


def kjo_net(X_in, keep_probability, num_of_classes):
    '''CNN Architecture.
    
    Args:
        X_in: tf.placeholder(tf.float32, (None, 32, 32, 1)), preprocessed images.
        keep_probability: tf.placeholder(tf.float32), probability of keeping nodes
            during dropout.
    
    Returns:
        logits
    '''
    # Hyperparameters used with tf.truncated_normal to set random
    # initial values of the weights and biases in each layer.
    mu = 0.0  # Default 0.0
    sigma = 0.1  # Default 0.1
    #
    # If using ReLU activations, initialize with slight positive bias, e.g., 0.1,
    # to avoid "dead neurons". 
    initial_bias = 0.05
    
    
    # Layer 1: Convolutional, 32x32x1 --> 28x28x16.
    # In nominal LeNet was 32x32x1 --> 28x28x6.
    # shape := (
    #   filter_height,
    #   filter_width,
    #   input_depth aka input_channel_cnt,
    #   output_depth aka output_channel_cnt).
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 16), mean=mu, stddev=sigma),name='conv1_W')
    conv1_b = tf.Variable(tf.zeros(16) + initial_bias, name='conv1_b')
    conv1 = tf.nn.conv2d(X_in, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)  # Activation.
    #
    # Pooling. 28x28x16 --> 14x14x16.
    # In nominal LeNet was 28x28x6 --> 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')

    
    # Layer 2: Convolutional, 14x14x16 --> 10x10x32.
    # In nominal LeNet was 14x14x6 --> 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 32), mean=mu, stddev=sigma),name='conv2_W')
    conv2_b = tf.Variable(tf.zeros(32) + initial_bias, name='conv2_b')
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)  # Activation.
    #
    # Pooling, 10x10x32 --> 5x5x32.
    # In nominal LeNet was 10x10x16 --> 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    
    # Flatten layer, 5x5x32 --> 800.
    # In nominal LeNet was 5x5x16 --> 400.
    # Flattens tensor into 2 dimensions (batches, length).
    fc0 = flatten(conv2)
    
    
    # Layer 3: Fully Connected, 800 --> 600.
    # In nominal LeNet was 400 --> 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(800, 600), mean=mu, stddev=sigma),name='fc1_W')
    fc1_b = tf.Variable(tf.zeros(600) + initial_bias, name='fc1_b')
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)  # Activation.
    #
    # Dropout. 
    fc1 = tf.nn.dropout(fc1, keep_probability)
    
    
    # Layer 4: Fully Connected, 600 --> 500.
    # In nominal LeNet was 120 --> 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(600, 500), mean=mu, stddev=sigma),name='fc2_W')
    fc2_b = tf.Variable(tf.zeros(500) + initial_bias, name='fc2_b')
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.relu(fc2)  # Activation.
    # Dropout. 
    fc2 = tf.nn.dropout(fc2, keep_probability)
    
    
    # Layer 5: Fully Connected, 500 --> class_cnt (default 43).
    # In nominal LeNet was 84 --> 10.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(500, num_of_classes), mean=mu, stddev=sigma),name='fc3_W')
    fc3_b = tf.Variable(tf.zeros(num_of_classes) + initial_bias, name='fc3_b')
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits