import tensorflow as tf
import tensorflow.contrib.layers as slim
import layers

ACTIVATION = tf.nn.relu
DATA_FORMAT = 'NCHW'

class SRN:
    def __init__(self, config=None):
        # format parameters
        self.dtype = tf.float32
        self.input_range = 2 # internal range of input. 1: [0,1], 2: [-1,1]
        self.output_range = 2 # internal range of output. 1: [0,1], 2: [-1,1]
        self.data_format = DATA_FORMAT
        self.in_channels = 3
        self.out_channels = 3
        self.input_shape = [None] * 4
        self.output_shape = [None] * 4
        self.input_shape[-3 if self.data_format == 'NCHW' else -1] = self.in_channels
        self.output_shape[-3 if self.data_format == 'NCHW' else -1] = self.out_channels
        # train parameters
        self.random_seed = None
        self.var_ema = 0.999
        # generator parameters
        self.generator_acti = ACTIVATION
        self.generator_wd = 1e-6
        self.generator_lr = 1e-3
        self.generator_lr_step = 500
        self.generator_vkey = 'generator_var'
        self.generator_lkey = 'generator_loss'
        # copy all the properties from config object
        if config is not None:
            self.__dict__.update(config.__dict__)
        # create a moving average object for trainable variables
        if self.var_ema > 0:
            self.ema = tf.train.ExponentialMovingAverage(self.var_ema)
        # state
        self.training = tf.Variable(False, trainable=False, name='training')

    def ResBlock(self, last, channels, kernel=3, stride=1, biases=True, format=DATA_FORMAT,
        dilate=1, activation=ACTIVATION, normalizer=None,
        initializer=None, regularizer=None, collections=None):
        biases = tf.initializers.zeros(self.dtype) if biases else None
        if initializer is None:
            initializer = tf.initializers.variance_scaling(
                1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        skip = last
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            dilate, activation, normalizer, None, initializer, regularizer, biases,
            variables_collections=collections)
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            dilate, None, normalizer, None, initializer, regularizer, biases,
            variables_collections=collections)
        last += skip
        return last

    def EBlock(self, last, channels, kernel=3, stride=2, format=DATA_FORMAT,
        activation=ACTIVATION, regularizer=None, collections=None):
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            1, activation, weights_initializer=initializer,
            weights_regularizer=regularizer, variables_collections=collections)
        last = self.ResBlock(last, channels, format=format,
            activation=activation, regularizer=regularizer, collections=collections)
        last = self.ResBlock(last, channels, format=format,
            activation=activation, regularizer=regularizer, collections=collections)
        return last

    def DBlock(self, last, channels, channels2, kernel=3, stride=2, format=DATA_FORMAT,
        activation=ACTIVATION, regularizer=None, collections=None):
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        last = self.ResBlock(last, channels, format=format,
            activation=activation, regularizer=regularizer, collections=collections)
        last = self.ResBlock(last, channels, format=format,
            activation=activation, regularizer=regularizer, collections=collections)
        last = slim.conv2d_transpose(last, channels2, kernel, stride, 'SAME', format,
            activation, weights_initializer=initializer,
            weights_regularizer=regularizer, variables_collections=collections)
        return last

    def OutBlock(self, last, channels, channels2, kernel=3, stride=1, format=DATA_FORMAT,
        activation=ACTIVATION, regularizer=None, collections=None):
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        last = self.ResBlock(last, channels, format=format,
            activation=activation, regularizer=regularizer, collections=collections)
        last = self.ResBlock(last, channels, format=format,
            activation=activation, regularizer=regularizer, collections=collections)
        last = slim.conv2d(last, channels2, kernel, stride, 'SAME', format,
            1, None, weights_initializer=initializer,
            weights_regularizer=regularizer, variables_collections=collections)
        return last

    def generator(self, last):
        # parameters
        main_scope = 'generator'
        format = self.data_format
        activation = self.generator_acti
        regularizer = slim.l2_regularizer(self.generator_wd)
        var_key = self.generator_vkey
        # model definition
        skips = []
        with tf.variable_scope(main_scope):
            skips.append(last)
            with tf.variable_scope('InBlock'):
                last = self.EBlock(last, 32, 3, 1, format, activation,
                    regularizer, var_key)
                skips.append(last)
            with tf.variable_scope('EBlock_1'):
                last = self.EBlock(last, 64, 3, 2, format, activation,
                    regularizer, var_key)
                skips.append(last)
            with tf.variable_scope('EBlock_2'):
                last = self.EBlock(last, 128, 3, 2, format, activation,
                    regularizer, var_key)
            with tf.variable_scope('DBlock_1'):
                last = self.DBlock(last, 128, 64, 3, 2, format, activation,
                    regularizer, var_key)
                last += skips.pop()
            with tf.variable_scope('DBlock_2'):
                last = self.DBlock(last, 64, 32, 3, 2, format, activation,
                    regularizer, var_key)
                last += skips.pop()
            with tf.variable_scope('OutBlock'):
                last = self.OutBlock(last, 32, self.out_channels, 3, 1, format, activation,
                    regularizer, var_key)
                last += skips.pop()
        # trainable/model/save/restore variables
        self.g_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=main_scope)
        self.g_mvars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope=main_scope)
        self.g_mvars = [i for i in self.g_mvars if i not in self.g_tvars]
        self.g_svars = list(set(self.g_tvars + self.g_mvars))
        self.g_rvars = self.g_svars.copy()
        # restore moving average of trainable variables
        if self.var_ema > 0:
            self.g_rvars = {**{self.ema.average_name(var): var for var in self.g_tvars},
                **{var.op.name: var for var in self.g_mvars}}
        return last

    def build_g_loss(self, ref, pred):
        loss_key = self.generator_lkey
        with tf.variable_scope(loss_key):
            summaries = []
            # data range conversion
            if self.output_range == 2:
                ref = (ref + 1) * 0.5
                pred = (pred + 1) * 0.5
            # L1 loss
            l1_loss = tf.losses.absolute_difference(ref, pred, 1.0, 'l1_loss')
            summaries.append(tf.summary.scalar('l1_loss', l1_loss))
            # total loss
            losses = tf.losses.get_losses(loss_key)
            total_loss = tf.add_n(losses, 'total_loss')
            # regularization loss
            reg_losses = tf.losses.get_regularization_losses('generator')
            reg_loss = tf.add_n(reg_losses, 'reg_loss')
            tf.summary.scalar('reg_loss', reg_loss)
            # final loss
            self.g_loss = total_loss + reg_loss
            tf.summary.scalar('g_loss', self.g_loss)
        return summaries

    def build_model(self, inputs=None):
        # inputs
        if inputs is None:
            self.inputs = tf.placeholder(self.dtype, self.input_shape, name='Input')
        else:
            self.inputs = tf.identity(inputs, name='Input')
            self.inputs.set_shape(self.input_shape)
        if self.input_range == 2:
            self.inputs = self.inputs * 2 - 1
        # forward pass
        self.outputs = self.generator(self.inputs)
        # outputs
        if self.output_range == 2:
            tf.multiply(self.outputs + 1, 0.5, name='Output')
        else:
            tf.identity(self.outputs, name='Output')
        # all the restore variables
        self.rvars = self.g_rvars
        # return outputs
        return self.outputs

    def build_train(self, inputs=None, labels=None):
        # reference outputs
        if labels is None:
            self.labels = tf.placeholder(self.dtype, self.output_shape, name='Label')
        else:
            self.labels = tf.identity(labels, name='Label')
            self.labels.set_shape(self.output_shape)
        if self.output_shape == 2:
            self.labels = self.labels * 2 - 1
        # build model
        self.build_model(inputs)
        # build generator loss
        g_summaries = self.build_g_loss(self.labels, self.outputs)
        # summary
        loss_summary = tf.summary.merge(g_summaries)
        # return total loss
        return self.g_loss, loss_summary

    def train(self, global_step):
        # dependencies to be updated
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # learning rate
        g_lr = tf.train.cosine_decay(self.generator_lr, global_step, self.generator_lr_step)
        tf.summary.scalar('generator_lr', g_lr)
        # optimizer
        g_opt = tf.contrib.opt.NadamOptimizer(g_lr)
        with tf.control_dependencies(update_ops):
            g_grads_vars = g_opt.compute_gradients(self.g_loss, self.g_tvars)
            update_ops = [g_opt.apply_gradients(g_grads_vars, global_step)]
        # histogram for gradients and variables
        for grad, var in g_grads_vars:
            tf.summary.histogram(var.op.name + '/grad', grad)
            tf.summary.histogram(var.op.name, var)
        # save moving average of trainalbe variables
        if self.var_ema > 0:
            with tf.control_dependencies(update_ops):
                update_ops = [self.ema.apply(self.g_tvars)]
            self.g_svars = [self.ema.average(var) for var in self.g_tvars] + self.g_mvars
        # all the saver variables
        self.svars = self.g_svars
        # return training op
        with tf.control_dependencies(update_ops):
            g_train_op = tf.no_op('g_train')
        return g_train_op
