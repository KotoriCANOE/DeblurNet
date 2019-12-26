from abc import ABCMeta, abstractmethod
import tensorflow as tf
import tensorflow.contrib.layers as slim
import layers

ACTIVATION = layers.Swish
DATA_FORMAT = 'NCHW'

# Generator

class GeneratorConfig:
    def __init__(self):
        # format parameters
        self.dtype = tf.float32
        self.data_format = DATA_FORMAT
        self.in_channels = 3
        self.out_channels = 3
        # model parameters
        self.biases = False
        self.activation = ACTIVATION
        self.normalization = None
        self.res_kernel = [3, 3]
        self.scaling = 1
        # train parameters
        self.random_seed = 0
        self.var_ema = 0.999
        self.weight_decay = 0#1e-6

class GeneratorBase(GeneratorConfig):
    __metaclass__ = ABCMeta

    def __init__(self, name='Generator', config=None):
        super().__init__()
        self.name = name
        self.tvars = None
        self.mvars = None
        self.svars = None
        self.rvars = None
        # copy all the properties from config object
        if config is not None:
            self.__dict__.update(config.__dict__)
        # create a moving average object for trainable variables
        if self.var_ema > 0:
            self.ema = tf.train.ExponentialMovingAverage(self.var_ema)

    def ResBlock(self, last, kernel=[3, 3], stride=[1, 1],
        biases=False, format=DATA_FORMAT,
        dilate=1, activation=ACTIVATION, normalizer=None,
        regularizer=None, collections=None):
        in_channels = last.shape.as_list()[-3 if format == 'NCHW' else -1]
        biases = tf.initializers.zeros(self.dtype) if biases else None
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'truncated_normal', self.random_seed, self.dtype)
        skip = last
        # pre-activation
        if normalizer: last = normalizer(last)
        if activation: last = activation(last)
        # convolution
        last = slim.conv2d(last, in_channels, kernel, stride, 'SAME', format,
            dilate, activation, normalizer, None, initializer, regularizer, biases,
            variables_collections=collections)
        last = slim.conv2d(last, in_channels, kernel, stride, 'SAME', format,
            dilate, None, None, None, initializer, regularizer, biases,
            variables_collections=collections)
        # skip connection
        last = layers.SEUnit(last, None, format, regularizer)
        last += skip
        return last

    def EBlock(self, last, channels, resblocks=1,
        kernel=[3, 3], stride=[2, 2], biases=True, format=DATA_FORMAT,
        activation=ACTIVATION, normalizer=None, regularizer=None, collections=None):
        # pre-activation
        if activation: last = activation(last)
        # down-convolution
        if stride[-1] > 1:
            last = layers.conv2d_downscale2d(last, channels, kernel[-1])
            # last = layers.conv2d_downscale2d(layers.blur2d(last), channels, kernel[-1])
        else:
            last = layers.conv2d(last, channels, kernel[-1])
        # bias
        if biases:
            last = slim.bias_add(last, variables_collections=collections, data_format=format)
        # residual blocks
        activation_res = ACTIVATION if activation is None else activation
        for i in range(resblocks):
            with tf.variable_scope('ResBlock_{}'.format(i)):
                last = self.ResBlock(last, self.res_kernel, biases=False, format=format,
                    activation=activation_res, normalizer=normalizer,
                    regularizer=regularizer, collections=collections)
        return last

    def DBlock(self, last, channels, resblocks=1,
        kernel=[3, 3], stride=[2, 2], biases=True, format=DATA_FORMAT,
        activation=ACTIVATION, normalizer=None, regularizer=None, collections=None):
        # residual blocks
        activation_res = ACTIVATION if activation is None else activation
        for i in range(resblocks):
            with tf.variable_scope('ResBlock_{}'.format(i)):
                last = self.ResBlock(last, self.res_kernel, biases=False, format=format,
                    activation=activation_res, normalizer=normalizer,
                    regularizer=regularizer, collections=collections)
        # pre-activation
        if activation: last = activation(last)
        # up-convolution
        if stride[-1] > 1:
            last = layers.upscale2d_conv2d(last, channels, kernel[-1])
            # last = layers.blur2d(layers.upscale2d_conv2d(last, channels, kernel[-1]))
        else:
            last = layers.conv2d(last, channels, kernel[-1])
        # bias
        if biases:
            last = slim.bias_add(last, variables_collections=collections, data_format=format)
        return last

    def Resize(self, last, stride, format=DATA_FORMAT):
        upscale = stride if isinstance(stride, int) else max(stride)
        if upscale <= 1:
            return last
        with tf.variable_scope(None, 'Upsample'):
            upsize = tf.shape(last)
            upsize = upsize[-2:] if format == 'NCHW' else upsize[-3:-1]
            upsize = upsize * stride[0:2]
            if format == 'NCHW':
                last = tf.transpose(last, (0, 2, 3, 1))
            last = tf.image.resize_nearest_neighbor(last, upsize)
            if format == 'NCHW':
                last = tf.transpose(last, (0, 3, 1, 2))
            return last

    def apply_ema(self, update_ops=[]):
        if not self.var_ema:
            return update_ops
        with tf.variable_scope('EMA'):
            with tf.control_dependencies(update_ops):
                update_ops = [self.ema.apply(self.tvars)]
            self.svars = [self.ema.average(var) for var in self.tvars] + self.mvars
        return update_ops

    @abstractmethod
    def def_model(self, last, activation, normalizer, regularizer):
        pass

    def __call__(self, last, reuse=None):
        format = self.data_format
        # function objects
        activation = self.activation
        if self.normalization == 'Batch':
            normalizer = lambda x: slim.batch_norm(x, 0.999, center=True, scale=True,
                is_training=self.training, data_format=format, renorm=False)
        elif self.normalization == 'Instance':
            normalizer = lambda x: slim.instance_norm(x, center=True, scale=True, data_format=format)
        elif self.normalization == 'Group':
            normalizer = lambda x: (slim.group_norm(x, x.shape.as_list()[-3] // 16, -3, (-2, -1))
                if format == 'NCHW' else slim.group_norm(x, x.shape.as_list()[-1] // 16, -1, (-3, -2)))
        else:
            normalizer = None
        regularizer = slim.l2_regularizer(self.weight_decay) if self.weight_decay else None
        # main model
        with tf.variable_scope(self.name, reuse=reuse):
            self.training = tf.Variable(False, trainable=False, name='training',
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MODEL_VARIABLES])
            last = self.def_model(last, activation, normalizer, regularizer)
        # trainable/model/save/restore variables
        self.tvars = tf.trainable_variables(self.name)
        self.mvars = tf.model_variables(self.name)
        self.mvars = [i for i in self.mvars if i not in self.tvars]
        self.svars = list(set(self.tvars + self.mvars))
        self.rvars = self.svars.copy()
        # restore moving average of trainable variables
        if self.var_ema > 0:
            with tf.variable_scope('EMA'):
                self.rvars = {**{self.ema.average_name(var): var for var in self.tvars},
                    **{var.op.name: var for var in self.mvars}}
        # return
        return last

class GeneratorResNet(GeneratorBase):
    def def_model(self, last, activation, normalizer, regularizer):
        format = self.data_format
        skip_connection = lambda x, y: x + y
        # skip stack
        skips = []
        # encoder
        with tf.variable_scope('InBlock'):
            skips.append(last)
            last = self.EBlock(last, 32, 0, [3, 3], [1, 1],
                True, format, None, None, regularizer)
        with tf.variable_scope('InResBlock'):
            skips.append(last)
            last = self.ResBlock(last, [3, 3], [1, 1],
                biases=self.biases, format=format,
                activation=activation, normalizer=normalizer,
                regularizer=regularizer)
            skips.append(last)
        # residual blocks
        resblocks = 8
        for depth in range(resblocks):
            with tf.variable_scope('ResBlock_{}'.format(depth)):
                last = self.ResBlock(last, [3, 3], [1, 1],
                    biases=self.biases, format=format,
                    activation=activation, normalizer=normalizer,
                    regularizer=regularizer)
        # decoder
        with tf.variable_scope('OutResBlock'):
            last = skip_connection(last, skips.pop())
            last = self.ResBlock(last, [3, 3], [1, 1],
                biases=self.biases, format=format,
                activation=activation, normalizer=normalizer,
                regularizer=regularizer)
            last = skip_connection(last, skips.pop())
        with tf.variable_scope('UpBlock'):
            if self.scaling > 1:
                last = self.DBlock(last, 32, 2, [3, 3], [self.scaling, self.scaling],
                    self.biases, format, activation, normalizer, regularizer)
        with tf.variable_scope('OutBlock'):
            last = self.DBlock(last, self.out_channels, 0, [3, 3], [1, 1],
                True, format, activation, normalizer, regularizer)
            if self.scaling == 1:
                last += skips.pop()
        with tf.variable_scope('SkipBlock'):
            if self.scaling > 1:
                skip = skips.pop()
                skip = self.DBlock(skip, self.out_channels, 0, [7, 7], [self.scaling, self.scaling],
                    False, format, None, None, regularizer)
                last += skip
        # return
        return last

class GeneratorSRN(GeneratorBase):
    def def_model(self, last, activation, normalizer, regularizer):
        format = self.data_format
        skip_connection = lambda x, y: x + y
        # skip stack
        skips = []
        # encoder
        with tf.variable_scope('InBlock'):
            skips.append(last)
            last = self.EBlock(last, 32, 0, [3, 3], [1, 1],
                True, format, None, None, regularizer)
        with tf.variable_scope('EBlock_0'):
            skips.append(last)
            last = self.EBlock(last, 32, 2, [3, 3], [1, 1],
                self.biases, format, activation, normalizer, regularizer)
        with tf.variable_scope('EBlock_1'):
            skips.append(last)
            last = self.EBlock(last, 64, 2, [3, 3], [2, 2],
                self.biases, format, activation, normalizer, regularizer)
        with tf.variable_scope('EBlock_2'):
            skips.append(last)
            last = self.EBlock(last, 96, 2, [3, 3], [2, 2],
                self.biases, format, activation, normalizer, regularizer)
        with tf.variable_scope('EBlock_3'):
            skips.append(last)
            last = self.EBlock(last, 128, 2, [3, 3], [2, 2],
                self.biases, format, activation, normalizer, regularizer)
        # decoder
        with tf.variable_scope('DBlock_3'):
            last = self.DBlock(last, 96, 2, [3, 3], [2, 2],
                self.biases, format, activation, normalizer, regularizer)
            last = skip_connection(last, skips.pop())
        with tf.variable_scope('DBlock_2'):
            last = self.DBlock(last, 64, 2, [3, 3], [2, 2],
                self.biases, format, activation, normalizer, regularizer)
            last = skip_connection(last, skips.pop())
        with tf.variable_scope('DBlock_1'):
            last = self.DBlock(last, 32, 2, [3, 3], [2, 2],
                self.biases, format, activation, normalizer, regularizer)
            last = skip_connection(last, skips.pop())
        with tf.variable_scope('DBlock_0'):
            last = self.DBlock(last, 32, 2, [3, 3], [1, 1],
                self.biases, format, activation, normalizer, regularizer)
            last = skip_connection(last, skips.pop())
        with tf.variable_scope('UpBlock'):
            if self.scaling > 1:
                last = self.DBlock(last, 32, 2, [3, 3], [self.scaling, self.scaling],
                    self.biases, format, activation, normalizer, regularizer)
        with tf.variable_scope('OutBlock'):
            last = self.DBlock(last, self.out_channels, 0, [3, 3], [1, 1],
                True, format, activation, normalizer, regularizer)
            if self.scaling == 1:
                last += skips.pop()
        with tf.variable_scope('SkipBlock'):
            if self.scaling > 1:
                skip = skips.pop()
                skip = self.DBlock(skip, self.out_channels, 0, [7, 7], [self.scaling, self.scaling],
                    False, format, None, None, regularizer)
                last += skip
        # return
        return last

class GeneratorResUNet(GeneratorBase):
    def DBlock(self, last, channels, resblocks=1,
        kernel=[3, 3], stride=[1, 1], biases=True, format=DATA_FORMAT,
        activation=ACTIVATION, normalizer=None, regularizer=None, collections=None):
        biases = tf.initializers.zeros(self.dtype) if biases else None
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'truncated_normal', self.random_seed, self.dtype)
        # residual blocks
        activation_res = ACTIVATION if activation is None else activation
        for i in range(resblocks):
            with tf.variable_scope('ResBlock_{}'.format(i)):
                last = self.ResBlock(last, self.res_kernel, biases=False, format=format,
                    activation=activation_res, normalizer=normalizer,
                    regularizer=regularizer, collections=collections)
        # pre-activation
        if activation: last = activation(last)
        # upsample
        last = self.Resize(last, stride, format)
        # convolution
        last = slim.conv2d(last, channels, kernel, [1, 1], 'SAME', format,
            1, None, None, weights_initializer=initializer,
            weights_regularizer=regularizer, biases_initializer=biases,
            variables_collections=collections)
        return last

    def def_model(self, last, activation, normalizer, regularizer):
        format = self.data_format
        skip_connection = lambda x, y: x + y
        # skip stack
        skips = []
        # encoder
        with tf.variable_scope('InBlock'):
            skips.append(last)
            last = self.EBlock(last, 32, 0, [3, 3], [1, 1],
                True, format, None, None, regularizer)
        with tf.variable_scope('EBlock_0'):
            skips.append(last)
            last = self.EBlock(last, 32, 2, [3, 3], [1, 1],
                self.biases, format, activation, normalizer, regularizer)
        with tf.variable_scope('EBlock_1'):
            skips.append(last)
            last = self.EBlock(last, 32, 2, [3, 3], [1, 1],
                self.biases, format, activation, normalizer, regularizer)
        # decoder
        with tf.variable_scope('DBlock_1'):
            last = self.DBlock(last, 32, 2, [3, 3], [1, 1],
                self.biases, format, activation, normalizer, regularizer)
            last = skip_connection(last, skips.pop())
        with tf.variable_scope('DBlock_0'):
            last = self.DBlock(last, 32, 2, [3, 3], [1, 1],
                self.biases, format, activation, normalizer, regularizer)
            last = skip_connection(last, skips.pop())
        with tf.variable_scope('OutBlock'):
            last = self.DBlock(last, self.out_channels, 0, [3, 3], [1, 1],
                True, format, activation, normalizer, regularizer)
            last = skip_connection(last, skips.pop())
        # return
        return last
