import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
from collections import Iterable

def Swish(last, scope=None):
    with tf.variable_scope(scope, 'Swish'):
        return last * tf.sigmoid(last)

def PReLU(last, format=None, collections=None, dtype=tf.float32, scope=None):
    if format is None:
        format = 'NHWC'
    shape = last.get_shape()
    shape = shape[-3 if format == 'NCHW' else -1]
    shape = [shape, 1, 1]
    with tf.variable_scope(scope, 'PReLU'):
        alpha = tf.get_variable('alpha', shape, dtype,
            tf.zeros_initializer(), collections=collections)
        if format == 'NCHW':
            alpha = tf.squeeze(alpha, axis=[-2, -1])
        last = tf.maximum(0.0, last) + alpha * tf.minimum(0.0, last)
    return last

def SEUnit(last, channels=None, format=None, collections=None, scope=None):
    in_channels = int(last.get_shape()[-3 if format == 'NCHW' else -1])
    if channels is None:
        channels = in_channels
    if format is None:
        format = 'NHWC'
    if collections is not None and not isinstance(collections, Iterable):
        collections = [collections]
    with tf.variable_scope(scope, 'SEUnit'):
        skip = last
        last = tf.reduce_mean(last, [-2, -1] if format == 'NCHW' else [-3, -2])
        last = slim.fully_connected(last, channels, tf.nn.relu,
            variables_collections=collections)
        last = slim.fully_connected(last, in_channels, tf.sigmoid)
        hw_idx = -1 if format == 'NCHW' else -2
        last = tf.expand_dims(tf.expand_dims(last, hw_idx), hw_idx)
        last = tf.multiply(skip, last)
    return last

def SmoothL1(labels, predictions, weights=1.0, scope=None, loss_collection=tf.GraphKeys.LOSSES):
    diff = predictions - labels
    absdiff = tf.abs(diff)
    smoothl1 = tf.cond(absdiff < 1, lambda: 0.5 * tf.square(diff), lambda: absdiff - 0.5)
    smoothl1 *= weights
    tf.losses.add_loss(smoothl1, loss_collection)
    return smoothl1

# Gaussian filter window for Conv2D
def GaussWindow(radius, sigma, channels=1, one_dim=False, dtype=tf.float32):
    if one_dim:
        y, x = np.mgrid[0 : 1, -radius : radius+1]
    else:
        # w = exp((x*x + y*y) / (-2.0*sigma*sigma))
        y, x = np.mgrid[-radius : radius+1, -radius : radius+1]
    w = -0.5 * (np.square(x) + np.square(y))
    w = w.reshape(list(w.shape) + [1, 1])
    # allow input a Tensor as sigma
    w = tf.constant(w, dtype=dtype)
    if not isinstance(sigma, tf.Tensor):
        sigma = tf.constant(sigma, dtype=dtype)
    g = tf.exp(w / tf.square(sigma))
    g /= tf.reduce_sum(g)
    # multi-channel
    if channels > 1:
        g = tf.concat([g] * channels, axis=-2)
    return g

# SS-SSIM/MS-SSIM implementation
# https://github.com/tensorflow/models/blob/master/compression/image_encoder/msssim.py
# https://stackoverflow.com/a/39053516
def SS_SSIM(img1, img2, ret_cs=False, mean_metric=True, radius=5, sigma=1.5, L=1, data_format='NHWC', one_dim=False, scope=None):
    with tf.variable_scope(scope, 'SS_SSIM'):
        # L: depth of image (255 in case the image has a differnt scale)
        window = GaussWindow(radius, sigma, one_dim=one_dim) # window shape [radius*2+1, radius*2+1]
        K1 = 0.01
        K2 = 0.03
        L_sq = L * L
        C1 = K1 * K1 * L_sq
        C2 = K2 * K2 * L_sq
        # implement
        mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID', data_format=data_format)
        mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1], padding='VALID', data_format=data_format)
        mu1_sq = tf.square(mu1)
        mu2_sq = tf.square(mu2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = tf.nn.conv2d(tf.square(img1), window, strides=[1,1,1,1], padding='VALID', data_format=data_format) - mu1_sq
        sigma2_sq = tf.nn.conv2d(tf.square(img2), window, strides=[1,1,1,1], padding='VALID', data_format=data_format) - mu2_sq
        sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1], padding='VALID', data_format=data_format) - mu1_mu2
        l_map = (2.0 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        cs_map = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        ssim_map = l_map * cs_map
        # metric
        if mean_metric:
            ssim_map = tf.reduce_mean(ssim_map)
            cs_map = tf.reduce_mean(cs_map)
        if ret_cs: value = (ssim_map, cs_map)
        else: value = ssim_map
    return value

def MS_SSIM(img1, img2, weights=None, radius=5, sigma=1.5, L=1, data_format='NHWC', one_dim=False, scope=None):
    with tf.variable_scope(scope, 'MS_SSIM'):
        if not weights:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        weights = tf.constant(weights, dtype=tf.float32)
        levels = weights.get_shape()[0].value
        mssim = []
        mcs = []
        # multi-scale
        if one_dim:
            window = [1,1,1,2] if data_format == 'NCHW' else [1,1,2,1]
        else:
            window = [1,1,2,2] if data_format == 'NCHW' else [1,2,2,1]
        for _ in range(levels):
            ssim, cs = SS_SSIM(img1, img2, ret_cs=True, mean_metric=True,
                radius=radius, sigma=sigma, L=L, data_format=data_format, one_dim=one_dim)
            mssim.append(tf.nn.relu(ssim)) # avoiding negative value
            mcs.append(tf.nn.relu(cs)) # avoiding negative value
            img1 = tf.nn.avg_pool(img1, window, window, padding='SAME', data_format=data_format)
            img2 = tf.nn.avg_pool(img2, window, window, padding='SAME', data_format=data_format)
        # list to tensor of dim D+1
        mcs = tf.stack(mcs, axis=0)
        value = tf.reduce_prod(mcs[0:levels - 1] ** weights[0:levels - 1]) * \
                              (mssim[levels - 1] ** weights[levels - 1])
    return value

# arXiv 1511.08861
def MS_SSIM2(img1, img2, radius=5, sigma=[0.5, 1, 2, 4, 8], L=1, norm=True, data_format='NHWC', one_dim=False, scope=None):
    with tf.variable_scope(scope, 'MS_SSIM2'):
        levels = len(sigma)
        mssim = []
        mcs = []
        for _ in range(levels):
            ssim, cs = SS_SSIM(img1, img2, ret_cs=True, mean_metric=False,
                radius=radius, sigma=sigma[_], L=L, data_format=data_format, one_dim=one_dim)
            mssim.append(tf.nn.relu(ssim)) # avoiding negative value
            mcs.append(tf.nn.relu(cs)) # avoiding negative value
        # list to tensor of dim D+1
        mcs = tf.stack(mcs, axis=0)
        value = tf.reduce_prod(mcs[0:levels - 1], axis=0) * mssim[levels - 1]
        value = tf.reduce_mean(value)
        if norm: value **= 1.0 / levels
    return value

def DiscriminatorLoss(real, fake, loss_type):
    loss_type = loss_type.lower()
    if 'wgan' in loss_type:
        real_loss = -tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)
    elif loss_type == 'lsgan':
        real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake))
    elif loss_type == 'gan' or loss_type == 'dragan':
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(fake), logits=fake))
    elif loss_type == 'hinge':
        real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real))
        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake))
    else:
        real_loss = 0
        fake_loss = 0
    loss = real_loss + fake_loss
    return loss

def GeneratorLoss(fake, loss_type):
    loss_type = loss_type.lower()
    if 'wgan' in loss_type:
        fake_loss = -tf.reduce_mean(fake)
    elif loss_type == 'lsgan':
        fake_loss = tf.reduce_mean(tf.squared_difference(fake, 1.0))
    elif loss_type == 'gan' or loss_type == 'dragan':
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(fake), logits=fake))
    elif loss_type == 'hinge':
        fake_loss = -tf.reduce_mean(fake)
    else:
        fake_loss = 0
    loss = fake_loss
    return loss
