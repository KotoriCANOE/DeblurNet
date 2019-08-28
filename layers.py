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

def SEUnit(last, channels=None, format=None, regularizer=None, collections=None, scope=None):
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
            weights_regularizer=regularizer,
            variables_collections=collections)
        last = slim.fully_connected(last, in_channels, tf.sigmoid,
            weights_regularizer=regularizer,
            variables_collections=collections)
        hw_idx = -1 if format == 'NCHW' else -2
        last = tf.expand_dims(tf.expand_dims(last, hw_idx), hw_idx)
        last = tf.multiply(skip, last)
    return last

def SmoothL1(labels, predictions, mean=True, weights=1.0, scope=None, loss_collection=tf.GraphKeys.LOSSES):
    with tf.variable_scope(scope, 'SmoothL1'):
        diff = predictions - labels
        absdiff = tf.abs(diff)
        condmask = tf.cast(absdiff < 1, tf.float32)
        smoothl1 = condmask * (0.5 * tf.square(diff)) + (1 - condmask) * (absdiff - 0.5)
        if mean:
            smoothl1 = tf.reduce_mean(smoothl1)
        if weights != 1.0:
            smoothl1 *= weights
        if mean and loss_collection is not None:
            tf.losses.add_loss(smoothl1, loss_collection)
        return smoothl1

# convert RGB to Y scale
def RGB2Y(last, data_format='NHWC', scope=None):
    with tf.variable_scope(scope, 'RGB2Y'):
        c1 = 1 / 3
        coef = [c1, c1, c1]
        t = tf.constant(coef, shape=[1, 3], dtype=last.dtype)
        if data_format == 'NCHW':
            last = tf.transpose(last, (0, 2, 3, 1))
        shape = tf.shape(last)
        last = tf.reshape(last, [tf.reduce_prod(shape[:-1]), 3])
        last = tf.matmul(last, t, transpose_b=True)
        if data_format == 'NCHW':
            shape = [shape[0], 1, shape[1], shape[2]]
        else:
            shape = [shape[0], shape[1], shape[2], 1]
        last = tf.reshape(last, shape)
    return last

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

### Scaling

def _blur2d(x, f=[1,2,1], normalize=True, flip=False, stride=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(stride, int) and stride >= 1

    # Finalize filter kernel.
    f = np.array(f, dtype=np.float32)
    if f.ndim == 1:
        f = f[:, np.newaxis] * f[np.newaxis, :]
    assert f.ndim == 2
    if normalize:
        f /= np.sum(f)
    if flip:
        f = f[::-1, ::-1]
    f = f[:, :, np.newaxis, np.newaxis]
    f = np.tile(f, [1, 1, int(x.shape[1]), 1])

    # No-op => early exit.
    if f.shape == (1, 1) and f[0,0] == 1:
        return x

    # Convolve using depthwise_conv2d.
    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)  # tf.nn.depthwise_conv2d() doesn't support fp16
    f = tf.constant(f, dtype=x.dtype, name='filter')
    strides = [1, 1, stride, stride]
    x = tf.nn.depthwise_conv2d(x, f, strides=strides, padding='SAME', data_format='NCHW')
    x = tf.cast(x, orig_dtype)
    return x

def _upscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Upscale using tf.tile().
    s = x.shape
    x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = tf.tile(x, [1, 1, 1, factor, 1, factor])
    x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x

def _downscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # 2x2, float32 => downscale using _blur2d().
    if factor == 2 and x.dtype == tf.float32:
        f = [np.sqrt(gain) / factor] * factor
        return _blur2d(x, f=f, normalize=False, stride=factor)

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Large factor => downscale using tf.nn.avg_pool().
    # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
    ksize = [1, 1, factor, factor]
    return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW')

#----------------------------------------------------------------------------
# High-level ops for manipulating 4D activation tensors.
# The gradients of these are meant to be as efficient as possible.

def blur2d(x, f=[1,2,1], normalize=True):
    with tf.variable_scope(None, 'Blur2D'):
        @tf.custom_gradient
        def func(x):
            y = _blur2d(x, f, normalize)
            @tf.custom_gradient
            def grad(dy):
                dx = _blur2d(dy, f, normalize, flip=True)
                return dx, lambda ddx: _blur2d(ddx, f, normalize)
            return y, grad
        return func(x)

def upscale2d(x, factor=2):
    with tf.variable_scope(None, 'Upscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _upscale2d(x, factor)
            @tf.custom_gradient
            def grad(dy):
                dx = _downscale2d(dy, factor, gain=factor**2)
                return dx, lambda ddx: _upscale2d(ddx, factor)
            return y, grad
        return func(x)

def downscale2d(x, factor=2):
    with tf.variable_scope(None, 'Downscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _downscale2d(x, factor)
            @tf.custom_gradient
            def grad(dy):
                dx = _upscale2d(dy, factor, gain=1/factor**2)
                return dx, lambda ddx: _downscale2d(ddx, factor)
            return y, grad
        return func(x)

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolutional or fully-connected layer.

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, lrmul=1):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, channels_in, channels_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable('weight', shape=shape, initializer=init) * runtime_coef

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense(x, channels, **kwargs):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    with tf.variable_scope(None, 'Dense'):
        w = get_weight([x.shape[1].value, channels], **kwargs)
        w = tf.cast(w, x.dtype)
        return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Convolutional layer.

def conv2d(x, channels, kernel, **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    with tf.variable_scope(None, 'Conv2D'):
        w = get_weight([kernel, kernel, x.shape[1].value, channels], **kwargs)
        w = tf.cast(w, x.dtype)
        return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Fused convolution + scaling.
# Faster and uses less memory than performing the operations separately.

def upscale2d_conv2d(x, channels, kernel, fused_scale='auto', **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    assert fused_scale in [True, False, 'auto']
    if fused_scale == 'auto':
        fused_scale = min(x.shape[2:]) * 2 >= 128

    # Not fused => call the individual ops directly.
    if not fused_scale:
        return conv2d(upscale2d(x), channels, kernel, **kwargs)

    with tf.variable_scope(None, 'UpscaleConv2D'):
        # Fused => perform both ops simultaneously using tf.nn.conv2d_transpose().
        w = get_weight([kernel, kernel, x.shape[1].value, channels], **kwargs)
        w = tf.transpose(w, [0, 1, 3, 2]) # [kernel, kernel, channels_out, channels_in]
        w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
        w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
        w = tf.cast(w, x.dtype)
        os = [tf.shape(x)[0], channels, x.shape[2] * 2, x.shape[3] * 2]
        return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

def conv2d_downscale2d(x, channels, kernel, fused_scale='auto', **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    assert fused_scale in [True, False, 'auto']
    if fused_scale == 'auto':
        fused_scale = min(x.shape[2:]) >= 128

    # Not fused => call the individual ops directly.
    if not fused_scale:
        return downscale2d(conv2d(x, channels, kernel, **kwargs))

    with tf.variable_scope(None, 'ConvDownscale2D'):
        # Fused => perform both ops simultaneously using tf.nn.conv2d().
        w = get_weight([kernel, kernel, x.shape[1].value, channels], **kwargs)
        w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
        w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
        w = tf.cast(w, x.dtype)
        return tf.nn.conv2d(x, w, strides=[1,1,2,2], padding='SAME', data_format='NCHW')
