import tensorflow.compat.v1 as tf
import numpy as np
import os
from utils import bool_argument, eprint, reset_random, create_session
from data import DataImage as Data
from model import Model

# class for training session
class Train:
    def __init__(self, config):
        self.debug = None
        self.random_seed = None
        self.device = None
        self.postfix = None
        self.pretrain_dir = None
        self.train_dir = None
        self.restore = None
        self.save_steps = None
        self.ckpt_period = None
        self.log_frequency = None
        self.log_file = None
        self.batch_size = None
        # dataset
        self.num_epochs = None
        self.max_steps = None
        # copy all the properties from config object
        self.config = config
        self.__dict__.update(config.__dict__)

    def initialize(self):
        # arXiv 1509.09308
        # a new class of fast algorithms for convolutional neural networks using Winograd's minimal filtering algorithms
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
        # create training directory
        if not self.restore:
            if os.path.exists(self.train_dir):
                eprint('Confirm removing {}\n[Y/n]'.format(self.train_dir))
                if input() != 'Y':
                    import sys
                    sys.exit()
                import shutil
                shutil.rmtree(self.train_dir, ignore_errors=True)
                eprint('Removed: ' + self.train_dir)
            if not os.path.exists(self.train_dir):
                os.makedirs(self.train_dir)
        # set deterministic random seed
        if self.random_seed is not None:
            reset_random(self.random_seed)

    def get_dataset(self):
        self.data = Data(self.config)
        self.epoch_steps = self.data.epoch_steps
        self.max_steps = self.data.max_steps
        # pre-computing validation set
        self.val_inputs = []
        self.val_labels = []
        for _inputs, _labels in self.data.gen_val():
            self.val_inputs.append(_inputs)
            self.val_labels.append(_labels)

    def build_graph(self):
        with tf.device(self.device):
            self.model = Model(self.config)
            self.model.build_train()
            self.global_step = tf.train.get_or_create_global_step()
            self.g_train_op = self.model.train_g(self.global_step)
            self.loss_summary, self.g_train_summary = self.model.get_summaries()

    def build_saver(self):
        # a Saver object to restore the variables with mappings
        # only for restoring from pre-trained model
        if self.pretrain_dir and not self.restore:
            self.saver_pt = tf.train.Saver(self.model.rvars)
        # a Saver object to save recent checkpoints
        self.saver_ckpt = tf.train.Saver(max_to_keep=24,
            save_relative_paths=True)
        # a Saver object to save the variables without mappings
        # used for saving checkpoints throughout the entire training progress
        self.saver = tf.train.Saver(self.model.svars,
            max_to_keep=1 << 16, save_relative_paths=True)
        # save the graph
        self.saver.export_meta_graph(os.path.join(self.train_dir, 'model.meta'),
            as_text=False, clear_devices=True, clear_extraneous_savers=True)

    def create_session(self):
        self.train_writer = tf.summary.FileWriter(self.train_dir + '/train',
            tf.get_default_graph(), max_queue=20, flush_secs=120)
        self.val_writer = tf.summary.FileWriter(self.train_dir + '/val')
        return create_session(debug=self.debug)

    def run_sess(self, sess, global_step, data_gen, options=None, run_metadata=None):
        from datetime import datetime
        import time
        epoch = global_step // self.epoch_steps
        last_step = global_step + 1 >= self.max_steps
        logging = last_step or (self.log_frequency > 0 and
            global_step % self.log_frequency == 0)
        # training - g train op
        _inputs, _labels = next(data_gen)
        feed_dict = {self.model.generator.training: True,
            'Input:0': _inputs, 'Label:0': _labels}
        fetches = [self.g_train_op, self.model.g_losses_acc]
        if logging:
            fetches += [self.g_train_summary]
            _, _, summary = sess.run(fetches, feed_dict, options, run_metadata)
            self.train_writer.add_summary(summary, global_step)
        else:
            sess.run(fetches, feed_dict, options, run_metadata)
        # training - log summary
        if logging:
            # loss summary
            fetches = [self.loss_summary] + self.model.g_log_losses
            train_ret = sess.run(fetches)
            self.train_writer.add_summary(train_ret[0], global_step)
            # logging
            time_current = time.time()
            duration = time_current - self.log_last
            self.log_last = time_current
            sec_batch = duration / self.log_frequency if self.log_frequency > 0 else 0
            samples_sec = self.batch_size / sec_batch
            train_log = ('{}: (train) epoch {}, step {}: losses: {}'
                ' ({:.1f} samples/sec, {:.3f} sec/batch)'
                .format(datetime.now(), epoch, global_step,
                    train_ret[1:], samples_sec, sec_batch))
            eprint(train_log)
        # validation
        if logging:
            for _inputs, _labels in zip(
                self.val_inputs, self.val_labels):
                feed_dict = {'Input:0': _inputs, 'Label:0': _labels}
                fetches = [self.model.g_losses_acc]
                sess.run(fetches, feed_dict)
            # loss summary
            fetches = [self.loss_summary] + self.model.g_log_losses
            val_ret = sess.run(fetches)
            self.val_writer.add_summary(val_ret[0], global_step)
            # logging
            val_log = ('{} (val) epoch {}, step {}: losses: {}'
                .format(datetime.now(), epoch, global_step, val_ret[1:]))
            eprint(val_log)
        # log result for the last step
        if self.log_file and last_step:
            last_log = ('epoch {}, step {}, losses: {}'
                .format(epoch, global_step, val_ret[1:]))
            with open(self.log_file, 'a', encoding='utf-8') as fd:
                fd.write('Training No.{}\n'.format(self.postfix))
                fd.write(self.train_dir + '\n')
                fd.write('{}\n'.format(datetime.now()))
                fd.write(last_log + '\n\n')

    def run(self, sess):
        import time
        # restore from checkpoint
        if self.restore and os.path.exists(os.path.join(self.train_dir, 'checkpoint')):
            latest_ckpt = tf.train.latest_checkpoint(self.train_dir, 'checkpoint')
            self.saver_ckpt.restore(sess, latest_ckpt)
        # otherwise, initialize from start
        else:
            initializers = (tf.initializers.global_variables(),
                tf.initializers.local_variables())
            sess.run(initializers)
        # restore pre-trained model
        if self.pretrain_dir:
            latest_ckpt = tf.train.latest_checkpoint(self.pretrain_dir, 'checkpoint')
            self.saver_pt.restore(sess, latest_ckpt)
        # profiler
        # profile_offset = -1
        profile_offset = 100 + self.log_frequency // 2
        profile_step = 10000
        builder = tf.profiler.ProfileOptionBuilder
        profiler = tf.profiler.Profiler(sess.graph)
        # initialization
        self.log_last = time.time()
        ckpt_last = time.time()
        # dataset generator
        global_step = tf.train.global_step(sess, self.global_step)
        data_gen = self.data.gen_main(global_step)
        # run training session
        while True:
            # global step
            global_step = tf.train.global_step(sess, self.global_step)
            if global_step >= self.max_steps:
                eprint('Training finished at step={}'.format(global_step))
                break
            # run session
            if global_step % profile_step == profile_offset:
                # profiling every few steps
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_meta = tf.RunMetadata()
                self.run_sess(sess, global_step, data_gen, options, run_meta)
                profiler.add_step(global_step, run_meta)
                # profile the parameters
                if global_step == profile_offset:
                    ofile = os.path.join(self.train_dir, 'parameters.log')
                    profiler.profile_name_scope(
                        builder(builder.trainable_variables_parameter())
                        .with_file_output(ofile).build())
                # profile the timing of model operations
                ofile = os.path.join(self.train_dir,
                    'time_and_memory_{:0>7}.log'.format(global_step))
                profiler.profile_operations(builder(builder.time_and_memory())
                    .with_file_output(ofile).build())
                # generate a timeline
                timeline = os.path.join(self.train_dir, 'timeline')
                profiler.profile_graph(builder(builder.time_and_memory())
                    .with_step(global_step).with_timeline_output(timeline).build())
            else:
                self.run_sess(sess, global_step, data_gen)
            # save checkpoints periodically or when training finished
            if self.ckpt_period > 0:
                time_current = time.time()
                if time_current - ckpt_last >= self.ckpt_period or global_step + 1 >= self.max_steps:
                    ckpt_last = time_current
                    self.saver_ckpt.save(sess, os.path.join(self.train_dir, 'model.ckpt'),
                        global_step, 'checkpoint')
            # save model every few steps
            if self.save_steps > 0 and global_step % self.save_steps == 0:
                self.saver.save(sess, os.path.join(self.train_dir,
                    'model_{:0>7}'.format(global_step)),
                    write_meta_graph=False, write_state=False)
        # auto detect problems and generate advice
        ALL_ADVICE = {
            'ExpensiveOperationChecker': {},
            'AcceleratorUtilizationChecker': {},
            'JobChecker': {},
            'OperationChecker': {}
        }
        profiler.advise(ALL_ADVICE)

    def __call__(self):
        self.initialize()
        self.get_dataset()
        with tf.Graph().as_default():
            self.build_graph()
            self.build_saver()
            with self.create_session() as sess:
                self.run(sess)

def main(argv=None):
    # arguments parsing
    import argparse
    argp = argparse.ArgumentParser(argv[0])
    # training parameters
    argp.add_argument('dataset')
    argp.add_argument('--val-dir') # only for packed dataset
    bool_argument(argp, 'debug', False)
    argp.add_argument('--num-epochs', type=int, default=24)
    argp.add_argument('--max-steps', type=int)
    argp.add_argument('--random-seed', type=int)
    argp.add_argument('--device', default='/gpu:0')
    argp.add_argument('--postfix', default='')
    argp.add_argument('--pretrain-dir', default='')
    argp.add_argument('--train-dir', default='./train{postfix}.tmp')
    argp.add_argument('--restore', action='store_true')
    argp.add_argument('--save-steps', type=int, default=5000)
    argp.add_argument('--ckpt-period', type=int, default=1200)
    argp.add_argument('--log-frequency', type=int, default=100)
    argp.add_argument('--log-file', default='train.log')
    argp.add_argument('--batch-size', type=int)
    argp.add_argument('--val-size', type=int, default=256)
    # data parameters
    argp.add_argument('--dtype', type=int, default=2)
    argp.add_argument('--in-channels', type=int, default=3)
    argp.add_argument('--out-channels', type=int, default=3)
    # pre-processing parameters
    Data.add_arguments(argp, False)
    # model parameters
    Model.add_arguments(argp)
    argp.add_argument('--scaling', type=int, default=1)
    # parse
    args = argp.parse_args(argv[1:])
    Data.parse_arguments(args)
    args.train_dir = args.train_dir.format(postfix=args.postfix)
    args.dtype = [tf.int8, tf.float16, tf.float32, tf.float64][args.dtype]
    # run training
    train = Train(args)
    train()

if __name__ == '__main__':
    import sys
    main(sys.argv)
