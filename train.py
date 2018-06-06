import tensorflow as tf
import numpy as np
import os
from utils import eprint, listdir_files, reset_random, create_session
from input import inputs, input_arguments
from model import SRN

# class for training session
class Train:
    def __init__(self, config):
        self.dataset = None
        self.num_epochs = None
        self.max_steps = None
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
                import shutil
                shutil.rmtree(self.train_dir)
                eprint('Removed: ' + self.train_dir)
            os.makedirs(self.train_dir)
        # set deterministic random seed
        if self.random_seed is not None:
            reset_random(self.random_seed)

    def get_dataset(self):
        files = listdir_files(self.dataset, filter_ext=['.jpeg', '.jpg', '.png'])
        # random shuffle
        import random
        random.shuffle(files)
        # validation set
        self.val_set = files[:self.val_size]
        files = files[self.val_size:]
        # training set
        self.epoch_steps = len(files) // self.batch_size
        self.epoch_size = self.epoch_steps * self.batch_size
        if self.max_steps is None:
            self.max_steps = self.epoch_steps * self.num_epochs
        else:
            self.num_epochs = (self.max_steps + self.epoch_steps - 1) // self.epoch_steps
            self.config.num_epochs = self.num_epochs
        self.train_set = files[:self.epoch_size]
        eprint('train set: {}\nepoch steps: {}\nnum epochs: {}\nmax steps: {}\n'
            .format(len(self.train_set), self.epoch_steps, self.num_epochs, self.max_steps))
        # pre-computing validation set
        with tf.Graph().as_default():
            from copy import deepcopy
            val_config = deepcopy(self.config)
            val_config.batch_size = self.val_size
            with tf.device('/cpu:0'):
                val_data = inputs(val_config, self.val_set, is_testing=True)
            with create_session() as sess:
                self.val_inputs, self.val_labels = sess.run(val_data)

    def build_train(self):
        with tf.device('/cpu:0'):
            self.inputs, self.labels = inputs(
                self.config, self.train_set, is_training=True)
        with tf.device(self.device):
            self.model = SRN(self.config)
            self.g_loss, self.g_loss_main = self.model.build_train(
                self.inputs, self.labels)
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

    def run_sess(self, sess, global_step):
        from datetime import datetime
        import time
        last_step = global_step + 1 >= self.max_steps
        epoch = global_step // self.epoch_steps
        # training
        feed_dict = {'training:0': True}
        if last_step or (self.log_frequency > 0 and
            global_step % self.log_frequency == 0):
            summary, g_loss, _ = sess.run((self.all_summary, self.g_loss,
                self.g_train_op), feed_dict)
            self.train_writer.add_summary(summary, global_step)
            time_current = time.time()
            duration = time_current - self.log_last
            self.log_last = time_current
            sec_batch = duration / self.log_frequency if self.log_frequency > 0 else 0
            samples_sec = self.batch_size / sec_batch
            train_log = '{}: epoch {}, step {}, train loss: {:.5} ({:.1f} samples/sec, {:.3f} sec/batch)'\
                .format(datetime.now(), epoch, global_step, g_loss, samples_sec, sec_batch)
            eprint(train_log)
        else:
            sess.run((self.g_train_op), feed_dict)
        # validation
        if last_step or (self.val_frequency > 0 and
            global_step % self.val_frequency == 0):
            feed_dict = {self.inputs: self.val_inputs,
                self.labels: self.val_labels}
            summary, g_loss_main = sess.run((self.loss_summary, self.g_loss_main), feed_dict)
            self.val_writer.add_summary(summary, global_step)
            val_log = '{}: epoch {}, step {}, val loss: {:.5}'\
                .format(datetime.now(), epoch, global_step, g_loss_main)
            eprint(val_log)
        # log result for the last step
        if self.log_file and last_step:
            last_log = 'epoch {}, step {}, train loss: {:.5}, val loss: {:.5}'\
                .format(epoch, global_step, g_loss, g_loss_main)
            with open(self.log_file, 'a', encoding='utf-8') as fd:
                fd.write('Training No.{}\n'.format(self.postfix))
                fd.write(self.train_dir + '\n')
                fd.write('{}\n'.format(datetime.now()))
                fd.write(last_log + '\n\n')

    def train(self, sess):
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
            sess.run(tf.global_variables_initializer())
        # initialization
        self.log_last = time.time()
        ckpt_last = time.time()
        # run training session
        while True:
            # global step
            global_step = tf.train.global_step(sess, self.global_step)
            if global_step >= self.max_steps:
                eprint('Training finished at step={}'.format(global_step))
                break
            # run session
            self.run_sess(sess, global_step)
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

    def __call__(self):
        self.initialize()
        self.get_dataset()
        with tf.Graph().as_default():
            self.build_train()
            self.build_saver()
            with self.train_session() as sess:
                self.train(sess)

def main(argv=None):
    # arguments parsing
    import argparse
    argp = argparse.ArgumentParser()
    # training parameters
    argp.add_argument('dataset')
    argp.add_argument('--num_epochs', type=int, default=24)
    argp.add_argument('--max_steps', type=int) # 255000
    argp.add_argument('--random_seed', type=int)
    argp.add_argument('--device', default='/gpu:0')
    argp.add_argument('--postfix', default='')
    argp.add_argument('--pretrain_dir', default='')
    argp.add_argument('--train_dir', default='./train{postfix}.tmp')
    argp.add_argument('--restore', action='store_true')
    argp.add_argument('--save_steps', type=int, default=5000)
    argp.add_argument('--ckpt_period', type=int, default=600)
    argp.add_argument('--log_frequency', type=int, default=100)
    argp.add_argument('--val_frequency', type=int, default=100)
    argp.add_argument('--log-file', default='train.log')
    argp.add_argument('--batch-size', type=int, default=16)
    argp.add_argument('--val-size', type=int, default=32)
    # data parameters
    argp.add_argument('--patch-height', type=int, default=192)
    argp.add_argument('--patch-width', type=int, default=192)
    # pre-processing parameters
    input_arguments(argp)
    # model parameters
    SRN.add_arguments(argp)
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
