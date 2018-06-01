import tensorflow as tf
import tensorflow.contrib.layers as slim
import layers

ACTIVATION = tf.nn.relu
FORMAT = 'NCHW'

class SRN:
    def __init__(self, config=None):
        # copy all the properties from config object
        if config is not None:
            self.__dict__.update(config.__dict__)
        self.dtype = tf.float32
        self.seed = None
        # parameters
        self.format = FORMAT
        self.out_channels = 3
        self.generator_acti = ACTIVATION
        self.generator_wd = 1e-6
        self.generator_lr = 1e-3
        self.generator_decs = 500
        self.generator_vkey = 'generator_var'
        self.generator_lkey = 'generator_loss'
    
    def ResBlock(self, last, channels, kernel=3, stride=1, biases=True, format=FORMAT,
        dilate=1, activation=ACTIVATION, normalizer=None,
        initializer=None, regularizer=None, collections=None):
        biases = tf.initializers.zeros(self.dtype) if biases else None
        if initializer is None:
            initializer = tf.initializers.variance_scaling(
                1.0, 'fan_in', 'normal', self.seed, self.dtype)
        skip = last
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            dilate, activation, normalizer, None, initializer, regularizer, biases,
            variables_collections=collections)
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            dilate, None, normalizer, None, initializer, regularizer, biases,
            variables_collections=collections)
        last += skip
        return last
    
    def EBlock(self, last, channels, kernel=3, stride=2, format=FORMAT,
        activation=ACTIVATION, regularizer=None, collections=None):
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            1, activation, weights_regularizer=regularizer, variables_collections=collections)
        last = self.ResBlock(last, channels, format=format,
            activation=activation, regularizer=regularizer, collections=collections)
        last = self.ResBlock(last, channels, format=format,
            activation=activation, regularizer=regularizer, collections=collections)
        return last

    def DBlock(self, last, channels, channels2, kernel=3, stride=2, format=FORMAT,
        activation=ACTIVATION, regularizer=None, collections=None):
        last = self.ResBlock(last, channels, format=format,
            activation=activation, regularizer=regularizer, collections=collections)
        last = self.ResBlock(last, channels, format=format,
            activation=activation, regularizer=regularizer, collections=collections)
        last = slim.conv2d_transpose(last, channels2, kernel, stride, 'SAME', format,
            activation, weights_regularizer=regularizer, variables_collections=collections)
        return last
    
    def OutBlock(self, last, channels, channels2, kernel=3, stride=1, format=FORMAT,
        activation=ACTIVATION, regularizer=None, collections=None):
        last = self.ResBlock(last, channels, format=format,
            activation=activation, regularizer=regularizer, collections=collections)
        last = self.ResBlock(last, channels, format=format,
            activation=activation, regularizer=regularizer, collections=collections)
        last = slim.conv2d(last, channels2, kernel, stride, 'SAME', format,
            1, None, weights_regularizer=regularizer, variables_collections=collections)
        return last

    def generator(self, last, train=False):
        format = self.format
        activation = self.generator_acti
        regularizer = slim.l2_regularizer(self.generator_wd)
        var_key = self.generator_vkey
        skips = []
        with tf.variable_scope('generator'):
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
        self.g_vars = tf.get_collection(var_key)
        return last

    def generator_loss(self, ref, pred):
        loss_key = self.generator_lkey
        with tf.variable_scope(loss_key):
            l1_loss = tf.losses.absolute_difference(ref, pred, 1.0, 'l1_loss')
            tf.summary.scalar('l1_loss', l1_loss)
            losses = tf.losses.get_losses(loss_key)
            total_loss = tf.add_n(losses, 'total_loss')
            reg_losses = tf.losses.get_regularization_losses('generator')
            reg_loss = tf.add_n(reg_losses, 'l2_reg_loss')
            tf.summary.scalar('l2_reg_loss', reg_loss)
            self.g_loss = total_loss + reg_loss
        print('losses:', losses)
        print('reg_losses:', reg_losses)
        print(total_loss)
        return self.g_loss

    def train(self, global_step):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # learning rate
        g_lr = tf.train.cosine_decay(self.generator_lr, global_step, self.generator_decs)
        tf.summary.scalar('generator_lr', g_lr)
        # optimizer
        g_opt = tf.contrib.opt.NadamOptimizer(g_lr)
        with tf.control_dependencies(update_ops):
            g_grads_vars = g_opt.compute_gradients(self.g_loss, self.g_vars)
            for grad, var in g_grads_vars:
                tf.summary.histogram(grad.name, grad)
                tf.summary.histogram(var.name, grad)
            update_ops = [g_opt.apply_gradients(g_grads_vars, global_step)]
        # return train_op
        with tf.control_dependencies(update_ops):
            return tf.no_op('train_op')
