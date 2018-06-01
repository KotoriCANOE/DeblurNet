import tensorflow as tf
from tensorflow.contrib import slim
from collections import Iterable

def Swish(last, scope='Swish'):
    with tf.name_scope(scope):
        return last * tf.sigmoid(last)

def PReLU(last, format=None, collections=None, dtype=tf.float32, scope='PReLU'):
    if format is None:
        format = 'NHWC'
    shape = last.get_shape()
    shape = shape[-3 if format == 'NCHW' else -1]
    shape = [shape, 1, 1]
    with tf.name_scope(scope):
        alpha = tf.get_variable('alpha', shape, dtype,
            tf.zeros_initializer(), collections=collections)
        if format == 'NCHW':
            alpha = tf.squeeze(alpha, axis=[-2, -1])
        last = tf.maximum(0.0, last) + alpha * tf.minimum(0.0, last)
    return last

def SEUnit(last, channels=None, format=None, collections=None, scope='SEUnit'):
    in_channels = last.get_shape()[-3 if format == 'NCHW' else -1]
    if channels is None:
        channels = in_channels
    if format is None:
        format = 'NHWC'
    if collections is not None and not isinstance(collections, Iterable):
        collections = [collections]
    with tf.name_scope(scope):
        skip = last
        last = tf.reduce_mean(last, [-2, -1] if format == 'NCHW' else [-3, -2])
        last = slim.fully_connected(last, channels, tf.nn.relu,
            variables_collections=collections)
        last = slim.fully_connected(last, in_channels, tf.sigmoid)
        hw_idx = -1 if format == 'NCHW' else -2
        last = tf.expand_dims(tf.expand_dims(last, hw_idx), hw_idx)
        last = tf.multiply(skip, last)
    return last

