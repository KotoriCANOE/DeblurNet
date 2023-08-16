import tensorflow.compat.v1 as tf
import numpy as np
from collections.abc import Iterable

################################################################

DATA_FORMAT = 'NCHW'

def channels_first(format):
    return format.lower() in ['channels_first', 'ncw', 'nchw', 'ncdhw']

def channels_last(format):
    return format.lower() in ['channels_last', 'nwc', 'nhwc', 'ndhwc']

def format_select(format, NCHW, NHWC):
    if channels_first(format):
        return NCHW
    elif channels_last(format):
        return NHWC
    else:
        raise ValueError('Unrecognized format: {}'.format(format))

################################################################

def Swish(last, scope=None):
    with tf.variable_scope(scope, 'Swish'):
        try:
            return tf.nn.swish(last)
        except Exception:
            return last * tf.nn.sigmoid(last)

def PReLU(last, format=None, collections=None, dtype=tf.float32, scope=None):
    if format is None:
        format = DATA_FORMAT
    shape = last.get_shape()
    shape = shape[format_select(format, 1, -1)]
    shape = [shape, 1, 1]
    with tf.variable_scope(scope, 'PReLU'):
        alpha = tf.get_variable('alpha', shape, dtype,
            tf.zeros_initializer(), collections=collections)
        if channels_first(format):
            alpha = tf.squeeze(alpha, axis=[-2, -1])
        last = tf.maximum(0.0, last) + alpha * tf.minimum(0.0, last)
    return last

def SEUnit(last, channels=None, format=None, scope=None):
    if format is None:
        format = DATA_FORMAT
    in_channels = int(last.shape[format_select(format, 1, -1)])
    if channels is None:
        channels = in_channels
    with tf.variable_scope(scope, 'SEUnit'):
        skip = last
        last = tf.reduce_mean(last, format_select(format, [-2, -1], [-3, -2]))
        last = dense(last, channels, tf.nn.relu)
        last = dense(last, in_channels, tf.nn.sigmoid)
        hw_idx = format_select(format, -1, -2)
        last = tf.expand_dims(tf.expand_dims(last, hw_idx), hw_idx)
        last = tf.multiply(skip, last)
    return last

def SmoothL1(labels, predictions, mean=True, weights=1.0, scope=None,
    loss_collection=tf.GraphKeys.LOSSES):
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
def RGB2Y(last, data_format=None, scope=None):
    if data_format is None:
        data_format = DATA_FORMAT
    with tf.variable_scope(scope, 'RGB2Y'):
        c1 = 1 / 3
        coef = [c1, c1, c1]
        t = tf.constant(coef, shape=[1, 3], dtype=last.dtype)
        if channels_first(data_format):
            last = tf.transpose(last, (0, 2, 3, 1))
        shape = tf.shape(last)
        last = tf.reshape(last, [tf.reduce_prod(shape[:-1]), 3])
        last = tf.matmul(last, t, transpose_b=True)
        if channels_first(data_format):
            shape = [shape[0], 1, shape[1], shape[2]]
        else:
            shape = [shape[0], shape[1], shape[2], 1]
        last = tf.reshape(last, shape)
    return last

################################################################

TRANSFER_TABLE = {
    'BT709': 1, # high precision
    'UNSPECIFIED': 2,
    'BT470_M': 4,
    'BT470_BG': 5,
    'BT601': 6, # high precision
    'ST170_M': 6,
    'ST240_M': 7,
    'LINEAR': 8,
    'LOG_100': 9,
    'LOG_316': 10,
    'IEC_61966_2_4': 11,
    'BT1361': 12,
    'IEC_61966_2_1': 13, # IEC-61966-2-1 standard of sRGB
    'SRGB': 113, # accurate sRGB with curve continuity and slope continuity
    'BT2020_10': 14, # high precision
    'BT2020_12': 15, # high precision
    'ST2084': 16,
    'ST428': 17,
    'ARIB_B67': 18
}

def TransferConvert(l2g, last, transfer=None, gamma=1.0, epsilon=1e-8, scope=None, scope_default='TransferConvert'):
    # transfer characteristics
    if isinstance(transfer, str):
        transfer = TRANSFER_TABLE[transfer.upper()]
    if transfer is None or transfer in [2, 8]:
        formula = 0
    elif transfer == 4:
        formula = 0
        gamma *= 2.2
    elif transfer == 5:
        formula = 0
        gamma *= 2.8
    elif transfer in [1, 6, 14, 15]:
        formula = 1
        power = 0.45
        power_rec = 1 / power
        slope = 4.500
        alpha = 1.099296826809442
        beta = 0.018053968510807
        k0 = beta * slope
    elif transfer == 7:
        formula = 1
        power = 0.45
        power_rec = 1 / power
        slope = 4.0
        alpha = 1.1115
        k0 = 0.0912
        beta = k0 / slope
    elif transfer == 13:
        formula = 1
        power = 1 / 2.4
        power_rec = 2.4 # 12 / 5
        slope = 12.92 # 323 / 25
        alpha = 1.055 # 211 / 200
        k0 = 0.04045
        beta = k0 / slope
    elif transfer == 113:
        formula = 1
        power = 1 / 2.4
        power_rec = 2.4
        slope = 12.9232102
        alpha = 1.055
        k0 = 11 / 280
        beta = k0 / slope
    elif transfer == 9:
        formula = 2
        alpha = 1 / 2
        beta = 0.01
    elif transfer == 10:
        formula = 2
        alpha = 2 / 5
        beta = np.sqrt(10) / 1000
    else:
        raise ValueError('transfer={} currently not supported'.format(transfer))
    with tf.variable_scope(scope, scope_default):
        # clipping
        if gamma != 1.0 or formula in [1, 2]:
            last = tf.clip_by_value(last, 0, 1)
        # apply transfer
        if formula == 0:
            pass
        elif formula == 1:
            if l2g:
                last = tf.where_v2(last < beta, slope * last,
                    alpha * (last + epsilon) ** power - (alpha - 1))
            else:
                last = tf.where_v2(last < k0, (1 / slope) * last,
                    ((1 / alpha) * (last + (alpha - 1))) ** power_rec)
        elif formula == 2:
            if l2g:
                # omit beta, instead using log(x + epsilon) and max(0, x)
                last = tf.math.maximum(0.0, 1.0 + (alpha / np.log(10)) * tf.math.log(last + epsilon))
            else:
                last = tf.math.pow(10.0, (1 / alpha) * (last - 1.0))
        # pure gamma adjustment
        if gamma == 2.0:
            last = tf.math.sqrt(last + epsilon) if l2g else tf.math.square(last)
        elif gamma == 0.5:
            last = tf.math.square(last) if l2g else tf.math.sqrt(last + epsilon)
        elif gamma > 1.0:
            last = (last + epsilon) ** (1 / gamma) if l2g else last ** gamma
        elif gamma < 1.0:
            last = last ** (1 / gamma) if l2g else (last + epsilon) ** gamma
    return last

def Linear2Gamma(last, transfer=None, gamma=1.0, epsilon=1e-8, scope=None):
    return TransferConvert(True, last, transfer, gamma, epsilon, scope, 'Linear2Gamma')

def Gamma2Linear(last, transfer=None, gamma=1.0, epsilon=1e-8, scope=None):
    return TransferConvert(False, last, transfer, gamma, epsilon, scope, 'Gamma2Linear')

################################################################

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
def SS_SSIM(img1, img2, ret_cs=False, mean_metric=True, radius=5, sigma=1.5, L=1,
    data_format=None, one_dim=False, scope=None):
    if data_format is None:
        data_format = DATA_FORMAT
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

def MS_SSIM(img1, img2, weights=None, radius=5, sigma=1.5, L=1,
    data_format=None, one_dim=False, scope=None):
    if data_format is None:
        data_format = DATA_FORMAT
    with tf.variable_scope(scope, 'MS_SSIM'):
        if not weights:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        weights = tf.constant(weights, dtype=tf.float32)
        levels = weights.get_shape()[0].value
        mssim = []
        mcs = []
        # multi-scale
        if one_dim:
            window = format_select(data_format, [1,1,1,2], [1,1,2,1])
        else:
            window = format_select(data_format, [1,1,2,2], [1,2,2,1])
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
def MS_SSIM2(img1, img2, radius=5, sigma=[0.5, 1, 2, 4, 8], L=1,
    norm=True, data_format=None, one_dim=False, scope=None):
    if data_format is None:
        data_format = DATA_FORMAT
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

################################################################

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

def FeatureMapMultiNoise(last, train=False, axis=[0, 2, 3], base=1.4, seed=None):
    shape = last.shape
    # shape mask
    mask_mul = [0, 0, 0, 0]
    mask_add = [1, 1, 1, 1]
    for i in axis:
        mask_mul[i] = 1
        mask_add[i] = 0
    # noise shape
    noise_shape = tf.shape(last) * mask_mul + mask_add
    # generate random noise
    noise = tf.random.truncated_normal(noise_shape, 0.0, 1.0, seed=seed)
    noise = tf.math.pow(base, noise)
    result = last * noise
    # return noised result only during training
    ret = tf.cond(train, lambda: result, lambda: last)
    ret.set_shape(shape)
    return ret

################################################################
### Conv layers with scaling

def _blur2d(x, f=[1,2,1], normalize=True, flip=False, stride=1, format=DATA_FORMAT):
    assert x.shape.ndims == 4
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
    f = np.tile(f, [1, 1, x.shape[format_select(format, 1, -1)].value, 1])

    # No-op => early exit.
    if f.shape == (1, 1) and f[0,0] == 1:
        return x

    # Convolve using depthwise_conv2d.
    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)  # tf.nn.depthwise_conv2d() doesn't support fp16
    f = tf.constant(f, dtype=x.dtype, name='filter')
    strides = format_select(format, [1, 1, stride, stride], [1, stride, stride, 1])
    x = tf.nn.depthwise_conv2d(x, f, strides=strides, padding='SAME', data_format=format)
    x = tf.cast(x, orig_dtype)
    return x

def _upscale2d(x, factor=2, gain=1, format=DATA_FORMAT):
    assert x.shape.ndims == 4
    assert isinstance(factor, int) and factor >= 1

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Upscale using tf.tile().
    shape = x.shape.as_list()
    s = tf.shape(x)
    if channels_first(format):
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        x.set_shape([shape[0], shape[1],
            shape[2] * factor if shape[2] else None,
            shape[3] * factor if shape[3] else None])
    else:
        x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
        x = tf.tile(x, [1, 1, factor, 1, factor, 1])
        x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
        x.set_shape([shape[0],
            shape[1] * factor if shape[1] else None,
            shape[2] * factor if shape[2] else None,
            shape[3]])
    return x

def _downscale2d(x, factor=2, gain=1, format=DATA_FORMAT):
    assert x.shape.ndims == 4
    assert isinstance(factor, int) and factor >= 1

    # 2x2, float32 => downscale using _blur2d().
    if factor == 2 and x.dtype == tf.float32:
        f = [np.sqrt(gain) / factor] * factor
        return _blur2d(x, f=f, normalize=False, stride=factor, format=format)

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Large factor => downscale using tf.nn.avg_pool().
    # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
    ksize = [1, 1, factor, factor]
    strides = format_select(format, [1, 1, factor, factor], [1, factor, factor, 1])
    return tf.nn.avg_pool(x, ksize=ksize, strides=strides, padding='VALID', data_format=format)

#----------------------------------------------------------------------------
# High-level ops for manipulating 4D activation tensors.
# The gradients of these are meant to be as efficient as possible.

def blur2d(x, f=[1,2,1], normalize=True, format=DATA_FORMAT):
    with tf.variable_scope(None, 'Blur2D'):
        @tf.custom_gradient
        def func(x):
            y = _blur2d(x, f, normalize, format=format)
            @tf.custom_gradient
            def grad(dy):
                dx = _blur2d(dy, f, normalize, flip=True, format=format)
                return dx, lambda ddx: _blur2d(ddx, f, normalize, format=format)
            return y, grad
        return func(x)

def upscale2d(x, factor=2, format=DATA_FORMAT):
    with tf.variable_scope(None, 'Upscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _upscale2d(x, factor, format=format)
            @tf.custom_gradient
            def grad(dy):
                dx = _downscale2d(dy, factor, gain=factor**2, format=format)
                return dx, lambda ddx: _upscale2d(ddx, factor, format=format)
            return y, grad
        return func(x)

def downscale2d(x, factor=2, format=DATA_FORMAT):
    with tf.variable_scope(None, 'Downscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _downscale2d(x, factor, format=format)
            @tf.custom_gradient
            def grad(dy):
                dx = _upscale2d(dy, factor, gain=1/factor**2, format=format)
                return dx, lambda ddx: _downscale2d(ddx, factor, format=format)
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
    return tf.get_variable('weight', shape=shape, initializer=init, trainable=True) * runtime_coef

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense(x, channels, activation=None, normalizer=None, bias=True, **kwargs):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    with tf.variable_scope(None, 'Dense'):
        w = get_weight([x.shape[-1].value, channels], **kwargs)
        w = tf.cast(w, x.dtype)
        y = tf.matmul(x, w)
        if normalizer is not None:
            y = normalizer(y)
        elif bias:
            init = tf.initializers.zeros()
            b = tf.get_variable('bias', shape=(channels,), initializer=init, trainable=True)
            b = tf.cast(b, x.dtype)
            y = tf.nn.bias_add(y, b)
        if activation is not None:
            y = activation(y)
        return y

#----------------------------------------------------------------------------
# Convolutional layer.

def conv2d(x, channels, kernel, format=DATA_FORMAT, **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    with tf.variable_scope(None, 'Conv2D'):
        w = get_weight([kernel, kernel, x.shape[1].value, channels], **kwargs)
        w = tf.cast(w, x.dtype)
        return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format=format)

#----------------------------------------------------------------------------
# Fused convolution + scaling.
# Faster and uses less memory than performing the operations separately.

def upscale2d_conv2d(x, channels, kernel, fused_scale='auto', format=DATA_FORMAT, **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    assert fused_scale in [True, False, 'auto']
    if fused_scale == 'auto':
        fused_scale = min(x.shape[2:]) * 2 >= 128

    # Not fused => call the individual ops directly.
    if not fused_scale:
        return conv2d(upscale2d(x, format=format), channels, kernel, format=format, **kwargs)

    with tf.variable_scope(None, 'UpscaleConv2D'):
        # Fused => perform both ops simultaneously using tf.nn.conv2d_transpose().
        w = get_weight([kernel, kernel, x.shape[1].value, channels], **kwargs)
        w = tf.transpose(w, [0, 1, 3, 2]) # [kernel, kernel, channels_out, channels_in]
        w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
        w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
        w = tf.cast(w, x.dtype)
        os = format_select(format,
            [tf.shape(x)[0], channels, x.shape[2] * 2, x.shape[3] * 2],
            [tf.shape(x)[0], x.shape[1] * 2, x.shape[2] * 2, channels])
        strides = format_select(format, [1, 1, 2, 2], [1, 2, 2, 1])
        return tf.nn.conv2d_transpose(x, w, os, strides=strides, padding='SAME', data_format=format)

def conv2d_downscale2d(x, channels, kernel, fused_scale='auto', format=DATA_FORMAT, **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    assert fused_scale in [True, False, 'auto']
    if fused_scale == 'auto':
        fused_scale = min(x.shape[2:]) >= 128

    # Not fused => call the individual ops directly.
    if not fused_scale:
        return downscale2d(conv2d(x, channels, kernel, format=format, **kwargs), format=format)

    with tf.variable_scope(None, 'ConvDownscale2D'):
        # Fused => perform both ops simultaneously using tf.nn.conv2d().
        w = get_weight([kernel, kernel, x.shape[1].value, channels], **kwargs)
        w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
        w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
        w = tf.cast(w, x.dtype)
        strides = format_select(format, [1, 1, 2, 2], [1, 2, 2, 1])
        return tf.nn.conv2d(x, w, strides=strides, padding='SAME', data_format=format)

################################################################

def ExpCosineRestartsDecay(lr, global_step, lr_step=1000, m_mul=0.9, alpha=0.1,
    exp_decay=0.998, warmup_cycle=0, fix_lr=False):
    if warmup_cycle > 0: # with warm up
        warmup_step = ((1 << warmup_cycle) - 1) * lr_step
        warmup_lr = (1 - alpha) * (m_mul ** warmup_cycle) + alpha
        # cosine restarts
        # fix_lr scales the base LR to make sure the highest LR (after warm up) equals the given LR
        lr_mul = 1.0 / warmup_lr if fix_lr else 1.0
        lr_mul = tf.train.cosine_decay_restarts(lr_mul,
            global_step, lr_step, t_mul=2.0, m_mul=m_mul, alpha=alpha)
        # warm up
        lr_mul = tf.cond(global_step >= warmup_step, lambda: lr_mul,
            lambda: tf.cast(global_step, tf.float32) / warmup_step * (1.0 if fix_lr else warmup_lr))
    else: # without warm up
        lr_mul = tf.train.cosine_decay_restarts(1.0,
            global_step, lr_step, t_mul=2.0, m_mul=m_mul, alpha=alpha)
    lr_mul = tf.train.exponential_decay(lr_mul, global_step, 1000, exp_decay)
    return lr * lr_mul

def PlateauDecay(lr, global_step, loss, factor=0.5, window=5000, min_delta=-0.005, min_mul=1e-4):
    with tf.variable_scope(None, 'PlateauDecay'):
        # initialize variables
        lr_mul = tf.Variable(1.0, trainable=False, name='LRMultiplier')
        loss_sum = tf.Variable(float('inf'), trainable=False, name='LossSum')
        loss_count = tf.Variable(1.0, trainable=False, name='LossCount')
        loss_mean = tf.Variable(float('inf'), trainable=False, name='LossMean')
        loss_last = tf.Variable(float('inf'), trainable=False, name='LossMeanLast')
        # LR multiplier
        def DecayJudge():
            # calculate mean
            new_loss_last = loss_last.assign(loss_mean, use_locking=True)
            with tf.control_dependencies([new_loss_last]):
                new_loss_mean = loss_mean.assign(loss_sum / loss_count, use_locking=True)
            with tf.control_dependencies([new_loss_mean]):
                clear_sum = loss_sum.assign(0.0, use_locking=True)
                clear_count = loss_count.assign(0.0, use_locking=True)
            with tf.control_dependencies([clear_sum, clear_count]):
                cal_mean = tf.no_op('CalculateMean')
            # decay judge
            with tf.control_dependencies([cal_mean]):
                print_op = tf.print('last loss mean: ', new_loss_last, '\nloss mean: ', new_loss_mean)
            with tf.control_dependencies([print_op]):
                decay_on = new_loss_mean * (1 + min_delta) > new_loss_last
            new_lr_mul = tf.cond(decay_on, lambda: tf.math.maximum(min_mul, lr_mul * factor), lambda: lr_mul)
            print_op = tf.print('new LR multiplier: ', new_lr_mul)
            with tf.control_dependencies([print_op]):
                return lr_mul.assign(new_lr_mul, use_locking=True)
        new_lr_mul = tf.cond(tf.equal(global_step % window, 0), DecayJudge, lambda: lr_mul)
        # accumulate to sum and count
        with tf.control_dependencies([new_lr_mul]):
            acc_sum = loss_sum.assign_add(loss, use_locking=True)
            acc_count = loss_count.assign_add(1.0, use_locking=True)
        with tf.control_dependencies([acc_sum, acc_count]):
            lr = lr * new_lr_mul
        # return decayed learning rate
        return lr

################################################################
