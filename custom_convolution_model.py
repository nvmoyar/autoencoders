import tensorflow as tf


def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides, padding):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    :padding: text indicating valid or same padding
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
   
    feature_map = x_tensor.get_shape()[3].value
    weight = tf.Variable(tf.random_normal([conv_ksize[0], conv_ksize[1], feature_map, conv_num_outputs], mean=0.0, stddev=0.1))
    bias = tf.Variable(tf.zeros([conv_num_outputs]))
    conv_layer = tf.nn.conv2d(x_tensor, weight, strides=[1, conv_strides[0], conv_strides[1], 1], padding=padding)
    conv_layer = tf.nn.bias_add(conv_layer, bias)
    conv_layer = tf.nn.relu(conv_layer)
    conv_layer = tf.nn.max_pool(conv_layer, 
                                ksize=[1, pool_ksize[0], pool_ksize[1], 1], strides=[1, pool_strides[0], pool_strides[1], 1],
                                padding=padding)
            
    return conv_layer



def encoding_conv_net(x):
    """
    A convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : return: Tensor that represents logits
    """
    
    conv_ksize = (2, 2)
    conv_strides = (2, 2)
    
    pool_ksize = (2, 2)
    pool_strides = (2, 2)
            
    conv_layer = conv2d_maxpool(x, 64, conv_ksize, conv_strides, pool_ksize, pool_strides, 'SAME')
    conv_layer = conv2d_maxpool(conv_layer, 32, conv_ksize, conv_strides, pool_ksize, pool_strides, 'SAME')
    conv_layer = conv2d_maxpool(conv_layer, 16, (1, 1), (1, 1), (1, 1), (1, 1), 'SAME')
    conv_layer = conv2d_maxpool(conv_layer, 8, (1, 1), (1, 1), (1, 1), (1, 1), 'SAME')
    encoded = conv2d_maxpool(conv_layer, 4, (2, 2), (2, 2), (2, 2), (1, 1), 'SAME')
    
    return encoded



def conv2d_upscale(encoded_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides, padding):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    :padding: text indicating valid or same padding
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
   
    feature_map = x_tensor.get_shape()[3].value
    weight = tf.Variable(tf.random_normal([conv_ksize[0], conv_ksize[1], feature_map, conv_num_outputs], mean=0.0, stddev=0.1))
    bias = tf.Variable(tf.zeros([conv_num_outputs]))
    conv_layer = tf.nn.conv2d(x_tensor, weight, strides=[1, conv_strides[0], conv_strides[1], 1], padding=padding)
    conv_layer = tf.nn.bias_add(conv_layer, bias)
    conv_layer = tf.nn.relu(conv_layer)
    conv_layer = tf.nn.max_pool(conv_layer, 
                                ksize=[1, pool_ksize[0], pool_ksize[1], 1], strides=[1, pool_strides[0], pool_strides[1], 1],
                                padding=padding)
            
    return conv_layer



def decoding_conv_net(x):
    """
    A convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : return: Tensor that represents logits
    """
    
    conv_ksize = (2, 2)
    conv_strides = (2, 2)
    
    pool_ksize = (2, 2)
    pool_strides = (2, 2)
            
    conv_layer = conv2d_maxpool(x, 64, conv_ksize, conv_strides, pool_ksize, pool_strides, 'SAME')
    conv_layer = conv2d_maxpool(conv_layer, 32, conv_ksize, conv_strides, pool_ksize, pool_strides, 'SAME')
    conv_layer = conv2d_maxpool(conv_layer, 16, (1, 1), (1, 1), (1, 1), (1, 1), 'SAME')
    conv_layer = conv2d_maxpool(conv_layer, 8, (1, 1), (1, 1), (1, 1), (1, 1), 'SAME')
    encoded = conv2d_maxpool(conv_layer, 4, (2, 2), (2, 2), (2, 2), (1, 1), 'SAME')
    
    return encoded


