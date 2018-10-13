import tensorflow as tf
import tensorflow.contrib.layers as slim
import layers

ACTIVATION = tf.nn.relu
#ACTIVATION = layers.Swish
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
        self.batch_size = None
        self.patch_height = None
        self.patch_width = None
        # train parameters
        self.random_seed = None
        self.var_ema = 0.999
        # generator parameters
        self.generator_acti = ACTIVATION
        self.generator_wd = 1e-6
        self.generator_lr = 1e-3
        self.generator_lr_step = 1000
        self.generator_vkey = 'generator_var'
        self.generator_lkey = 'generator_loss'
        # collections
        self.train_sums = []
        self.loss_sums = []
        # copy all the properties from config object
        if config is not None:
            self.__dict__.update(config.__dict__)
        # internal parameters
        self.input_shape = [None] * 4
        self.input_shape[-3 if self.data_format == 'NCHW' else -1] = self.in_channels
        self.output_shape = [None] * 4
        self.output_shape[-3 if self.data_format == 'NCHW' else -1] = self.out_channels
        # create a moving average object for trainable variables
        if self.var_ema > 0:
            self.ema = tf.train.ExponentialMovingAverage(self.var_ema)

    @staticmethod
    def add_arguments(argp):
        # format parameters
        argp.add_argument('--input-range', type=int, default=2)
        argp.add_argument('--output-range', type=int, default=2)
        # training parameters
        argp.add_argument('--var-ema', type=float, default=0.999)
        argp.add_argument('--generator-wd', type=float, default=1e-6)
        argp.add_argument('--generator-lr', type=float, default=1e-3)
        argp.add_argument('--generator-lr-step', type=int, default=1000)

    def ResBlock(self, last, channels, kernel=3, stride=1, biases=True, format=DATA_FORMAT,
        dilate=1, activation=ACTIVATION, normalizer=None,
        regularizer=None, collections=None):
        biases = tf.initializers.zeros(self.dtype) if biases else None
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        skip = last
        if normalizer: last = normalizer(last)
        if activation: last = activation(last)
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            dilate, activation, normalizer, None, initializer, regularizer, biases,
            variables_collections=collections)
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            dilate, None, None, None, initializer, regularizer, biases,
            variables_collections=collections)
        last += skip
        return last

    def InBlock(self, last, channels, kernel=3, stride=1, format=DATA_FORMAT,
        activation=ACTIVATION, normalizer=None, regularizer=None, collections=None):
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            1, None, None, weights_initializer=initializer,
            weights_regularizer=regularizer, variables_collections=collections)
        last = self.ResBlock(last, channels, format=format,
            activation=activation, normalizer=normalizer,
            regularizer=regularizer, collections=collections)
        last = self.ResBlock(last, channels, format=format,
            activation=activation, normalizer=normalizer,
            regularizer=regularizer, collections=collections)
        return last

    def EBlock(self, last, channels, kernel=3, stride=2, format=DATA_FORMAT,
        activation=ACTIVATION, normalizer=None, regularizer=None, collections=None):
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        if activation: last = activation(last)
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            1, None, None, weights_initializer=initializer,
            weights_regularizer=regularizer, variables_collections=collections)
        last = self.ResBlock(last, channels, format=format,
            activation=activation, normalizer=normalizer,
            regularizer=regularizer, collections=collections)
        last = self.ResBlock(last, channels, format=format,
            activation=activation, normalizer=normalizer,
            regularizer=regularizer, collections=collections)
        return last

    def DBlock(self, last, channels, channels2, kernel=3, stride=2, format=DATA_FORMAT,
        activation=ACTIVATION, normalizer=None, regularizer=None, collections=None):
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        last = self.ResBlock(last, channels, format=format,
            activation=activation, normalizer=normalizer,
            regularizer=regularizer, collections=collections)
        last = self.ResBlock(last, channels, format=format,
            activation=activation, normalizer=normalizer,
            regularizer=regularizer, collections=collections)
        if activation: last = activation(last)
        last = slim.conv2d_transpose(last, channels2, kernel, stride, 'SAME', format,
            None, None, weights_initializer=initializer,
            weights_regularizer=regularizer, variables_collections=collections)
        return last

    def OutBlock(self, last, channels, channels2, kernel=3, stride=1, format=DATA_FORMAT,
        activation=ACTIVATION, normalizer=None, regularizer=None, collections=None):
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        last = self.ResBlock(last, channels, format=format,
            activation=activation, normalizer=normalizer,
            regularizer=regularizer, collections=collections)
        last = self.ResBlock(last, channels, format=format,
            activation=activation, normalizer=normalizer,
            regularizer=regularizer, collections=collections)
        if activation: last = activation(last)
        last = slim.conv2d(last, channels2, kernel, stride, 'SAME', format,
            1, None, None, weights_initializer=initializer,
            weights_regularizer=regularizer, variables_collections=collections)
        return last

    def generator(self, last):
        # parameters
        main_scope = 'generator'
        format = self.data_format
        var_key = self.generator_vkey
        # model definition
        with tf.variable_scope(main_scope):
            # states
            self.g_training = tf.Variable(False, trainable=False, name='training',
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MODEL_VARIABLES])
            # function objects
            activation = self.generator_acti
            normalizer = lambda x: slim.batch_norm(x, 0.999, center=True, scale=True,
                is_training=self.g_training, data_format=format, renorm=False)
            regularizer = slim.l2_regularizer(self.generator_wd)
            skips = []
            # network
            last = tf.identity(last, 'inputs')
            skips.append(last)
            with tf.variable_scope('InBlock'):
                last = self.InBlock(last, 32, 3, 1, format, activation,
                    normalizer, regularizer, var_key)
                skips.append(last)
            with tf.variable_scope('EBlock_1'):
                last = self.EBlock(last, 64, 3, 2, format, activation,
                    normalizer, regularizer, var_key)
                skips.append(last)
            with tf.variable_scope('EBlock_2'):
                last = self.EBlock(last, 128, 3, 2, format, activation,
                    normalizer, regularizer, var_key)
            with tf.variable_scope('DBlock_1'):
                last = self.DBlock(last, 128, 64, 3, 2, format, activation,
                    normalizer, regularizer, var_key)
                last += skips.pop()
            with tf.variable_scope('DBlock_2'):
                last = self.DBlock(last, 64, 32, 3, 2, format, activation,
                    normalizer, regularizer, var_key)
                last += skips.pop()
            with tf.variable_scope('OutBlock'):
                last = self.OutBlock(last, 32, self.out_channels, 3, 1, format, activation,
                    normalizer, regularizer, var_key)
                last += skips.pop()
            last = tf.identity(last, 'outputs')
        # trainable/model/save/restore variables
        self.g_tvars = tf.trainable_variables(main_scope)
        self.g_mvars = tf.model_variables(main_scope)
        self.g_mvars = [i for i in self.g_mvars if i not in self.g_tvars]
        self.g_svars = list(set(self.g_tvars + self.g_mvars))
        self.g_rvars = self.g_svars.copy()
        # restore moving average of trainable variables
        if self.var_ema > 0:
            with tf.variable_scope('variables_ema'):
                self.g_rvars = {**{self.ema.average_name(var): var for var in self.g_tvars},
                    **{var.op.name: var for var in self.g_mvars}}
        return last

    def build_g_loss(self, ref, pred):
        self.g_log_losses = []
        update_ops = []
        loss_key = self.generator_lkey
        with tf.variable_scope(loss_key):
            # L1 loss
            l1_loss = tf.losses.absolute_difference(ref, pred, 1.0)
            update_ops.append(self.loss_summary('l1_loss', l1_loss, self.g_log_losses))
            # total loss
            losses = tf.losses.get_losses(loss_key)
            g_main_loss = tf.add_n(losses, 'g_main_loss')
            # regularization loss
            g_reg_losses = tf.losses.get_regularization_losses('generator')
            g_reg_loss = tf.add_n(g_reg_losses)
            update_ops.append(self.loss_summary('g_reg_loss', g_reg_loss))
            # final loss
            self.g_loss = g_main_loss + g_reg_loss
            update_ops.append(self.loss_summary('g_loss', self.g_loss))
            # accumulate operator
            with tf.control_dependencies(update_ops):
                self.g_losses_acc = tf.no_op('g_losses_accumulator')

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
            self.outputs = tf.multiply(self.outputs + 1, 0.5, name='Output')
        else:
            self.outputs = tf.identity(self.outputs, name='Output')
        # all the saver variables
        self.svars = self.g_svars
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
        # build model
        self.build_model(inputs)
        # build generator loss
        self.build_g_loss(self.labels, self.outputs)
        # return total loss
        return self.g_loss

    def train(self, global_step):
        # saving memory with gradient checkpoints
        #self.set_reuse_checkpoints()
        #g_ckpt = tf.get_collection('checkpoints', 'generator')
        # dependencies to be updated
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # learning rate
        g_lr = tf.train.cosine_decay_restarts(self.generator_lr,
            global_step, self.generator_lr_step)
        self.train_sums.append(tf.summary.scalar('generator_lr', g_lr))
        # optimizer
        g_opt = tf.contrib.opt.NadamOptimizer(g_lr)
        with tf.control_dependencies(update_ops):
            #g_grads_vars = self.compute_gradients(self.g_loss, self.g_tvars, g_ckpt)
            g_grads_vars = g_opt.compute_gradients(self.g_loss, self.g_tvars)
            update_ops = [g_opt.apply_gradients(g_grads_vars, global_step)]
        # histogram for gradients and variables
        for grad, var in g_grads_vars:
            self.train_sums.append(tf.summary.histogram(var.op.name + '/grad', grad))
            self.train_sums.append(tf.summary.histogram(var.op.name, var))
        # save moving average of trainalbe variables
        if self.var_ema > 0:
            with tf.variable_scope('variables_ema'):
                with tf.control_dependencies(update_ops):
                    update_ops = [self.ema.apply(self.g_tvars)]
                self.g_svars = [self.ema.average(var) for var in self.g_tvars] + self.g_mvars
        # all the saver variables
        self.svars = self.g_svars
        # return training op
        with tf.control_dependencies(update_ops):
            g_train_op = tf.no_op('g_train')
        return g_train_op

    def loss_summary(self, name, loss, collection=None):
        with tf.variable_scope('loss_summary/' + name):
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
                self.loss_sums.append(tf.summary.scalar(name, loss_mean))
            # return after updating sum and count
            with tf.control_dependencies([acc_sum, acc_count]):
                return tf.identity(loss, 'loss')

    def get_summaries(self):
        train_summary = tf.summary.merge(self.train_sums)
        loss_summary = tf.summary.merge(self.loss_sums)
        return train_summary, loss_summary

    @staticmethod
    def set_reuse_checkpoints():
        import re
        # https://stackoverflow.com/a/36893840
        # get the name of all the tensors output by weight operations
        # MatMul, Conv2D, etc.
        graph = tf.get_default_graph()
        nodes = graph.as_graph_def().node
        regex = re.compile(r'^.+(?:MatMul|Conv2D|conv2d_transpose|BiasAdd)$')
        op_names = [n.name for n in nodes if re.match(regex, n.name)]
        tensors = [graph.get_tensor_by_name(n + ':0') for n in op_names]
        # add these tensors to collection 'checkpoints'
        for tensor in tensors:
            tf.add_to_collection('checkpoints', tensor)

    @staticmethod
    def compute_gradients(loss, var_list, checkpoints='collection'):
        # https://github.com/openai/gradient-checkpointing
        from memory_saving_gradients import gradients
        grads = gradients(loss, var_list, checkpoints=checkpoints)
        grads_vars = list(zip(grads, var_list))
        return grads_vars
