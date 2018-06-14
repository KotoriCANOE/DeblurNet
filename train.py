import tensorflow as tf
import numpy as np
import os
from utils import eprint, listdir_files, reset_random, create_session
from data import Data
from model import SRN

# class for training session
class Train:
    def __init__(self, config):
        self.random_seed = None
        self.device = None
        self.postfix = None
        self.pretrain_dir = None
        self.train_dir = None
        self.restore = None
        self.save_steps = None
        self.ckpt_period = None
        self.log_frequency = None
        self.val_frequency = None
        self.log_file = None
        self.batch_size = None
        self.val_size = None
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
                shutil.rmtree(self.train_dir)
                eprint('Removed: ' + self.train_dir)
            os.makedirs(self.train_dir)
        # set deterministic random seed
        if self.random_seed is not None:
            reset_random(self.random_seed)

    def get_dataset(self):
        self.data = Data(self.config)
        self.epoch_steps = self.data.epoch_steps
        self.max_steps = self.data.max_steps
        self.val_inputs = []
        self.val_labels = []
        for _inputs, _labels in self.data.get_val():
            self.val_inputs.append(_inputs)
            self.val_labels.append(_labels)

    def build_graph(self):
        with tf.device(self.device):
            self.model = SRN(self.config)
            self.g_loss = self.model.build_train()
            self.global_step = tf.train.get_or_create_global_step()
            self.g_train_op = self.model.train(self.global_step)
            self.all_summary, self.loss_summary = self.model.get_summaries()

    def build_saver(self):
        # a Saver object to restore the variables with mappings
        # only for restoring from pre-trained model
        if self.pretrain_dir and not self.restore:
            self.saver_pt = tf.train.Saver(self.model.rvars)
        # a Saver object to save recent checkpoints
        self.saver_ckpt = tf.train.Saver(max_to_keep=5,
            save_relative_paths=True)
        # a Saver object to save the variables without mappings
        # used for saving checkpoints throughout the entire training progress
        self.saver = tf.train.Saver(self.model.svars,
            max_to_keep=1 << 16, save_relative_paths=True)
        # save the graph
        self.saver.export_meta_graph(os.path.join(self.train_dir, 'model.meta'),
            as_text=False, clear_devices=True, clear_extraneous_savers=True)

    def train_session(self):
        self.train_writer = tf.summary.FileWriter(self.train_dir + '/train',
            tf.get_default_graph(), max_queue=20, flush_secs=120)
        self.val_writer = tf.summary.FileWriter(self.train_dir + '/val')
        return create_session()

    def run_sess(self, sess, global_step, data_gen, options=None, run_metadata=None):
        from datetime import datetime
        import time
        last_step = global_step + 1 >= self.max_steps
        epoch = global_step // self.epoch_steps
        # training - train op
        inputs, labels = next(data_gen)
        feed_dict = {self.model.g_training: True,
            'Input:0': inputs, 'Label:0': labels}
        fetch = (self.g_train_op, self.model.g_losses_acc)
        sess.run(fetch, feed_dict, options, run_metadata)
        # training - log summary
        if self.log_frequency > 0 and global_step % self.log_frequency == 0:
            fetch = [self.all_summary] + self.model.g_losses
            summary, train_loss = sess.run(fetch, feed_dict)
            self.train_writer.add_summary(summary, global_step)
            time_current = time.time()
            duration = time_current - self.log_last
            self.log_last = time_current
            sec_batch = duration / self.log_frequency if self.log_frequency > 0 else 0
            samples_sec = self.batch_size / sec_batch
            train_log = '{}: epoch {}, step {}, train loss: {:.5} ({:.1f} samples/sec, {:.3f} sec/batch)'\
                .format(datetime.now(), epoch, global_step,
                    train_loss, samples_sec, sec_batch)
            eprint(train_log)
        # validation
        if last_step or (self.val_frequency > 0 and
            global_step % self.val_frequency == 0):
            for inputs, labels in zip(self.val_inputs, self.val_labels):
                feed_dict = {'Input:0': inputs, 'Label:0': labels}
                fetch = [self.model.g_losses_acc]
                sess.run(fetch, feed_dict)
            fetch = [self.loss_summary] + self.model.g_losses
            summary, val_loss = sess.run(fetch, feed_dict)
            self.val_writer.add_summary(summary, global_step)
            val_log = '{}: epoch {}, step {}, val loss: {:.5}'\
                .format(datetime.now(), epoch, global_step, val_loss)
            eprint(val_log)
        # log result for the last step
        if self.log_file and last_step:
            last_log = 'epoch {}, step {}, train loss: {:.5}, val loss: {:.5}'\
                .format(epoch, global_step, train_loss, val_loss)
            with open(self.log_file, 'a', encoding='utf-8') as fd:
                fd.write('Training No.{}\n'.format(self.postfix))
                fd.write(self.train_dir + '\n')
                fd.write('{}\n'.format(datetime.now()))
                fd.write(last_log + '\n\n')

    def run(self, sess):
        import time
        # restore from checkpoint
        if self.restore and os.path.exists(os.path.join(self.train_dir, 'checkpoint')):
            lastest_ckpt = tf.train.latest_checkpoint(self.train_dir, 'checkpoint')
            self.saver_ckpt.restore(sess, lastest_ckpt)
        # restore pre-trained model
        elif self.pretrain_dir:
            self.saver_pt.restore(sess, os.path.join(self.pretrain_dir, 'model'))
        # otherwise, initialize from start
        else:
            initializers = (tf.initializers.global_variables(),
                tf.initializers.local_variables())
            sess.run(initializers)
        # profiler
        profile_offset = 1000 + self.log_frequency // 2
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
            with self.train_session() as sess:
                self.run(sess)

def main(argv=None):
    # arguments parsing
    import argparse
    argp = argparse.ArgumentParser()
    # training parameters
    argp.add_argument('dataset')
    argp.add_argument('--num-epochs', type=int, default=24)
    argp.add_argument('--max-steps', type=int)
    argp.add_argument('--random-seed', type=int)
    argp.add_argument('--device', default='/gpu:0')
    argp.add_argument('--postfix', default='')
    argp.add_argument('--pretrain-dir', default='')
    argp.add_argument('--train-dir', default='./train{postfix}.tmp')
    argp.add_argument('--restore', action='store_true')
    argp.add_argument('--save-steps', type=int, default=5000)
    argp.add_argument('--ckpt-period', type=int, default=600)
    argp.add_argument('--log-frequency', type=int, default=100)
    argp.add_argument('--val-frequency', type=int, default=100)
    argp.add_argument('--log-file', default='train.log')
    argp.add_argument('--batch-size', type=int, default=32)
    argp.add_argument('--val-size', type=int, default=256)
    # data parameters
    argp.add_argument('--dtype', type=int, default=2)
    argp.add_argument('--data-format', default='NCHW')
    argp.add_argument('--patch-height', type=int, default=128)
    argp.add_argument('--patch-width', type=int, default=128)
    argp.add_argument('--in-channels', type=int, default=3)
    argp.add_argument('--out-channels', type=int, default=3)
    # pre-processing parameters
    Data.add_arguments(argp)
    # model parameters
    SRN.add_arguments(argp)
    argp.add_argument('--scaling', type=int, default=1)
    # parse
    args = argp.parse_args(argv)
    args.train_dir = args.train_dir.format(postfix=args.postfix)
    args.dtype = [tf.int8, tf.float16, tf.float32, tf.float64][args.dtype]
    # run training
    train = Train(args)
    train()

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
