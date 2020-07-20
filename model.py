import tensorflow.compat.v1 as tf
from tensorflow import contrib
import layers
from network import GeneratorSRN as Generator

DATA_FORMAT = 'NCHW'

class Model:
    def __init__(self, config=None):
        # format parameters
        self.dtype = tf.float32
        self.data_format = DATA_FORMAT
        self.input_range = 2 # internal range of input. 1: [0,1], 2: [-1,1]
        self.output_range = 2 # internal range of output. 1: [0,1], 2: [-1,1]
        self.transfer = 'SRGB'
        self.loss_transfer = 'SRGB'
        self.in_channels = 3
        self.out_channels = 3
        # train parameters
        self.learning_rate = None
        self.weight_decay = None
        # collections
        self.g_train_sums = []
        self.loss_sums = []
        # copy all the properties from config object
        self.config = config
        if config is not None:
            self.__dict__.update(config.__dict__)
        self.transfer = self.transfer.upper()
        self.loss_transfer = self.loss_transfer.upper()
        # internal parameters
        self.input_shape = [None, None, None, None]
        self.input_shape[-3 if self.data_format == 'NCHW' else -1] = self.in_channels
        self.output_shape = [None, None, None, None]
        self.output_shape[-3 if self.data_format == 'NCHW' else -1] = self.out_channels

    @staticmethod
    def add_arguments(argp):
        # format parameters
        argp.add_argument('--data-format', default='NCHW')
        argp.add_argument('--input-range', type=int, default=2)
        argp.add_argument('--output-range', type=int, default=2)
        argp.add_argument('--transfer', default='SRGB')
        argp.add_argument('--loss-transfer', default='SRGB')
        # training parameters
        argp.add_argument('--learning-rate', type=float, default=1e-3)
        argp.add_argument('--weight-decay', type=float, default=5e-5)
        argp.add_argument('--var-ema', type=float, default=0.999)
        argp.add_argument('--grad-clip', type=float, default=-0.2) # 0.2|-0.2

    def build_model(self, inputs=None):
        # inputs
        if inputs is None:
            self.inputs = tf.placeholder(self.dtype, self.input_shape, name='Input')
        else:
            self.inputs = tf.identity(inputs, name='Input')
            self.inputs.set_shape(self.input_shape)
        inputs = self.inputs
        # convert to linear
        inputs = layers.Gamma2Linear(inputs, self.transfer)
        if self.input_range == 2:
            inputs = inputs * 2 - 1
        # forward pass
        self.generator = Generator('Generator', self.config)
        outputs = self.generator(inputs, reuse=None)
        # outputs
        if self.output_range == 2:
            outputs = tf.tanh(outputs)
            outputs = tf.multiply(outputs + 1, 0.5)
        # convert to gamma
        self.outputs = layers.Linear2Gamma(outputs, self.transfer)
        self.outputs = tf.identity(self.outputs, name='Output')
        self.outputs_gamma = (self.outputs if self.transfer == self.loss_transfer
            else layers.Linear2Gamma(outputs, self.loss_transfer))
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
        # convert to gamma if it's linear
        if self.transfer != self.loss_transfer:
            labels = layers.Gamma2Linear(self.labels, self.transfer)
            self.labels_gamma = layers.Linear2Gamma(labels, self.loss_transfer)
        # build model
        self.build_model(inputs)
        # build losses
        self.build_g_loss(self.labels_gamma, self.outputs_gamma)

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
            # SSIM loss
            labelsY = layers.RGB2Y(labels, self.data_format)
            outputsY = layers.RGB2Y(outputs, self.data_format)
            ssim_loss = 1 - layers.MS_SSIM2(labelsY, outputsY, sigma=[1.5, 4.0, 10.0],
                L=1, norm=False, data_format=self.data_format)
            tf.losses.add_loss(ssim_loss * 0.1)
            update_ops.append(self.loss_summary('ssim_loss', ssim_loss, self.g_log_losses))
            # regularization loss
            reg_losses = tf.losses.get_regularization_losses('Generator')
            reg_loss = tf.add_n(reg_losses)
            # tf.losses.add_loss(reg_loss)
            update_ops.append(self.loss_summary('reg_loss', reg_loss))
            # final loss
            losses = tf.losses.get_losses(loss_key)
            self.g_loss = tf.add_n(losses, 'total_loss')
            update_ops.append(self.loss_summary('loss', self.g_loss))
            # accumulate operator
            with tf.control_dependencies(update_ops):
                self.g_losses_acc = tf.no_op('accumulator')

    def train_g(self, global_step):
        model = self.generator
        # dependencies to be updated
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'Generator')
        # learning rate
        # steps/decay: 2047000/0.999, 1023000/0.998
        lr_mul = layers.ExpCosineRestartsDecay(1.0, global_step, exp_decay=0.998, warmup_cycle=6) # cycle=6
        # lr_mul = layers.PlateauDecay(1.0, global_step, self.g_loss)
        lr = self.learning_rate * lr_mul
        wd = self.weight_decay * lr_mul
        self.g_train_sums.append(tf.summary.scalar('Generator/LR', lr))
        # optimizer
        opt = contrib.opt.AdamWOptimizer(wd, lr, beta1=0.9, beta2=0.999)
        with tf.control_dependencies(update_ops):
            grads_vars = opt.compute_gradients(self.g_loss, model.tvars)
        # gradient clipping
        with tf.variable_scope(None, 'GradientClipping'):
            _grads, _vars = zip(*grads_vars)
            global_norm = tf.linalg.global_norm(_grads)
            if self.grad_clip > 0: # hard clipping
                _grads, _ = tf.clip_by_global_norm(_grads, self.grad_clip, use_norm=global_norm)
            elif self.grad_clip < 0: # soft clipping
                grad_clip = -self.grad_clip
                scale = grad_clip / (global_norm + grad_clip)
                _grads = [grad * scale for grad in _grads]
            grads_vars = list(zip(_grads, _vars))
        self.g_train_sums.append(tf.summary.scalar('Generator/grad_global_norm', global_norm))
        update_ops = [opt.apply_gradients(grads_vars, global_step, decay_var_list=model.wdvars)]
        # histogram for gradients and variables
        for grad, var in grads_vars:
            self.g_train_sums.append(tf.summary.histogram(var.op.name + '/grad', grad))
            self.g_train_sums.append(tf.summary.histogram(var.op.name, var))
        # save moving average of trainable variables
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
        loss_summary = tf.summary.merge(self.loss_sums) if self.loss_sums else None
        g_train_summary = tf.summary.merge(self.g_train_sums) if self.g_train_sums else None
        return loss_summary, g_train_summary
