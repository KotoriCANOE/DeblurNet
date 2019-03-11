import tensorflow as tf
import layers
from network import Generator

DATA_FORMAT = 'NCHW'

class Model:
    def __init__(self, config=None):
        # format parameters
        self.dtype = tf.float32
        self.input_range = 2 # internal range of input. 1: [0,1], 2: [-1,1]
        self.output_range = 2 # internal range of output. 1: [0,1], 2: [-1,1]
        self.data_format = DATA_FORMAT
        self.in_channels = 3
        self.out_channels = 3
        # collections
        self.g_train_sums = []
        self.loss_sums = []
        # copy all the properties from config object
        self.config = config
        if config is not None:
            self.__dict__.update(config.__dict__)
        # internal parameters
        self.input_shape = [None, None, None, None]
        self.input_shape[-3 if self.data_format == 'NCHW' else -1] = self.in_channels
        self.output_shape = [None, None, None, None]
        self.output_shape[-3 if self.data_format == 'NCHW' else -1] = self.out_channels

    @staticmethod
    def add_arguments(argp):
        # format parameters
        argp.add_argument('--input-range', type=int, default=2)
        argp.add_argument('--output-range', type=int, default=2)
        # training parameters
        argp.add_argument('--var-ema', type=float, default=0.999)

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
        self.generator = Generator('Generator', self.config)
        self.outputs = self.generator(self.inputs, reuse=None)
        # outputs
        if self.output_range == 2:
            self.outputs = tf.multiply(self.outputs + 1, 0.5, name='Output')
        else:
            self.outputs = tf.identity(self.outputs, name='Output')
        # all the saver variables
        self.svars = self.generator.svars
        # all the restore variables
        self.rvars = self.generator.rvars
        # return outputs
        return self.outputs

    def build_train(self, inputs=None, labels=None):
        # reference outputs
        if labels is None:
            self.labels = tf.placeholder(self.dtype, self.output_shape, name='Label')
        else:
            self.labels = tf.identity(labels, name='Label')
            self.labels.set_shape(self.output_shape)
        # build model
        self.build_model(inputs)
        # build generator loss
        self.build_g_loss(self.labels, self.outputs)
        # return total loss
        return self.g_loss

    def build_g_loss(self, labels, outputs):
        self.g_log_losses = []
        update_ops = []
        loss_key = 'GeneratorLoss'
        with tf.variable_scope(loss_key):
            # L1 loss
            l1_loss = tf.losses.absolute_difference(labels, outputs, 1.0,
                loss_collection=None)
            tf.losses.add_loss(l1_loss)
            update_ops.append(self.loss_summary('l1_loss', l1_loss, self.g_log_losses))
            # total loss
            losses = tf.losses.get_losses(loss_key)
            main_loss = tf.add_n(losses, 'main_loss')
            # regularization loss
            reg_losses = tf.losses.get_regularization_losses('Generator')
            reg_loss = tf.add_n(reg_losses)
            update_ops.append(self.loss_summary('reg_loss', reg_loss))
            # final loss
            self.g_loss = main_loss + reg_loss
            update_ops.append(self.loss_summary('loss', self.g_loss))
            # accumulate operator
            with tf.control_dependencies(update_ops):
                self.g_losses_acc = tf.no_op('accumulator')

    def train_g(self, global_step):
        model = self.generator
        # dependencies to be updated
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'Generator')
        # learning rate
        lr_base = 1e-3
        lr_step = 1000
        lr = tf.train.cosine_decay_restarts(lr_base,
            global_step, lr_step, t_mul=2.0, m_mul=0.9, alpha=1e-1)
        lr = tf.train.exponential_decay(lr, global_step, 1000, 0.997)
        self.g_train_sums.append(tf.summary.scalar('Generator/LR', lr))
        # optimizer
        opt = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999)
        with tf.control_dependencies(update_ops):
            grads_vars = opt.compute_gradients(self.g_loss, model.tvars)
            update_ops = [opt.apply_gradients(grads_vars, global_step)]
        # histogram for gradients and variables
        for grad, var in grads_vars:
            self.g_train_sums.append(tf.summary.histogram(var.op.name + '/grad', grad))
            self.g_train_sums.append(tf.summary.histogram(var.op.name, var))
        # save moving average of trainalbe variables
        update_ops = model.apply_ema(update_ops)
        # all the saver variables
        self.svars = model.svars
        # return training op
        with tf.control_dependencies(update_ops):
            train_op = tf.no_op('train_g')
        return train_op

    def loss_summary(self, name, loss, collection=None):
        with tf.variable_scope('LossSummary/' + name):
            # internal variables
            loss_sum = tf.get_variable('sum', (), tf.float32, tf.initializers.zeros(tf.float32))
            loss_count = tf.get_variable('count', (), tf.float32, tf.initializers.zeros(tf.float32))
            # accumulate to sum and count
            acc_sum = loss_sum.assign_add(loss, True)
            acc_count = loss_count.assign_add(1.0, True)
            # calculate mean
            loss_mean = tf.divide(loss_sum, loss_count, 'mean')
            if collection is not None:
                collection.append(loss_mean)
            # reset sum and count
            with tf.control_dependencies([loss_mean]):
                clear_sum = loss_sum.assign(0.0, True)
                clear_count = loss_count.assign(0.0, True)
            # log summary
            with tf.control_dependencies([clear_sum, clear_count]):
                self.loss_sums.append(tf.summary.scalar('value', loss_mean))
            # return after updating sum and count
            with tf.control_dependencies([acc_sum, acc_count]):
                return tf.identity(loss, 'loss')

    def get_summaries(self):
        g_train_summary = tf.summary.merge(self.g_train_sums) if self.g_train_sums else None
        loss_summary = tf.summary.merge(self.loss_sums) if self.loss_sums else None
        return g_train_summary, loss_summary
