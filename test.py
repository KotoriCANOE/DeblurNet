import tensorflow.compat.v1 as tf
import numpy as np
import os
from utils import bool_argument, eprint, listdir_files, reset_random, create_session, BatchPNG
from data import DataImage as Data
from model import Model

# losses measured for testing
def test_losses(ref, pred):
    # RGB color space
    RGB_mse = tf.losses.mean_squared_error(ref, pred, weights=1.0)
    RGB_mad = tf.losses.absolute_difference(ref, pred, weights=1.0)
    # return each loss
    return RGB_mse, RGB_mad

# class for testing session
class Test:
    def __init__(self, config):
        self.random_seed = None
        self.device = None
        self.postfix = None
        self.train_dir = None
        self.test_dir = None
        self.log_file = None
        self.batch_size = None
        # copy all the properties from config object
        self.config = config
        self.__dict__.update(config.__dict__)

    def initialize(self):
        # arXiv 1509.09308
        # a new class of fast algorithms for convolutional neural networks using Winograd's minimal filtering algorithms
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
        # create testing directory
        if not os.path.exists(self.train_dir):
            raise FileNotFoundError('Could not find folder {}'.format(self.train_dir))
        if os.path.exists(self.test_dir):
            eprint('Confirm removing {}\n[Y/n]'.format(self.test_dir))
            if input() != 'Y':
                import sys
                sys.exit()
            import shutil
            shutil.rmtree(self.test_dir, ignore_errors=True)
            eprint('Removed: ' + self.test_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        # set deterministic random seed
        if self.random_seed is not None:
            reset_random(self.random_seed)

    def get_dataset(self):
        self.data = Data(self.config)
        self.epoch_steps = self.data.epoch_steps
        self.max_steps = self.data.max_steps
        # pre-computing testing set
        self.test_inputs = []
        self.test_labels = []
        for _inputs, _labels in self.data.gen_main():
            self.test_inputs.append(_inputs)
            self.test_labels.append(_labels)

    def build_graph(self):
        with tf.device(self.device):
            inputs = tf.placeholder(tf.float32, name='inputs')
            labels = tf.placeholder(tf.float32, name='labels')
            self.model = Model(self.config)
            outputs = self.model.build_model(inputs)
            self.losses = list(test_losses(labels, outputs))
        # post-processing for output
        with tf.device('/cpu:0'):
            # convert to NHWC format
            if self.config.data_format == 'NCHW':
                inputs = tf.transpose(inputs, [0,2,3,1])
                labels = tf.transpose(labels, [0,2,3,1])
                outputs = tf.transpose(outputs, [0,2,3,1])
            # PNG output
            self.pngs = (BatchPNG(inputs, self.batch_size)
                + BatchPNG(labels, self.batch_size)
                + BatchPNG(outputs, self.batch_size))

    def build_saver(self):
        # a Saver object to restore the variables with mappings
        self.saver = tf.train.Saver(self.model.rvars)

    def run_last(self, sess):
        # latest checkpoint
        ckpt = tf.train.latest_checkpoint(self.train_dir)
        self.saver.restore(sess, ckpt)
        # to be fetched
        fetch = self.losses + self.pngs
        losses_sum = [0 for _ in range(len(self.losses))]
        # run session
        for step in range(self.epoch_steps):
            feed_dict = {'inputs:0': self.test_inputs[step],
                'labels:0': self.test_labels[step]}
            ret = sess.run(fetch, feed_dict)
            ret_losses = ret[0:len(self.losses)]
            ret_pngs = ret[len(self.losses):]
            # sum of losses
            for i in range(len(self.losses)):
                losses_sum[i] += ret_losses[i]
            # save images
            _start = step * self.batch_size
            _stop = _start + self.batch_size
            _range = range(_start, _stop)
            ofiles = (['{:0>5}.0.inputs.png'.format(i) for i in _range]
                + ['{:0>5}.1.labels.png'.format(i) for i in _range]
                + ['{:0>5}.2.outputs{}.png'.format(i, self.postfix) for i in _range])
            ofiles = [os.path.join(self.test_dir, f) for f in ofiles]
            for i in range(len(ret_pngs)):
                with open(ofiles[i], 'wb') as fd:
                    fd.write(ret_pngs[i])
        # summary
        if self.log_file:
            from datetime import datetime
            losses_mean = [l / self.epoch_steps for l in losses_sum]
            psnr = 10 * np.log10(1 / losses_mean[0]) if losses_mean[0] > 0 else 100
            test_log = 'PSNR (RGB):{}, MAD (RGB): {}'\
                .format(psnr, *losses_mean[1:])
            with open(self.log_file, 'a', encoding='utf-8') as fd:
                fd.write('Testing No.{}\n'.format(self.postfix))
                fd.write(self.test_dir + '\n')
                fd.write('{}\n'.format(datetime.now()))
                fd.write(test_log + '\n\n')

    def run_steps(self, sess):
        import re
        prefix = 'model_'
        # get checkpoints of every few steps
        ckpts = listdir_files(self.train_dir, recursive=False,
            filter_ext=['.index'])
        ckpts = [os.path.splitext(f)[0] for f in ckpts if prefix in f]
        ckpts.sort()
        stats = []
        # test all the checkpoints
        for ckpt in ckpts:
            self.saver.restore(sess, ckpt)
            # to be fetched
            fetch = self.losses
            losses_sum = [0 for _ in range(len(self.losses))]
            # run session
            for step in range(self.epoch_steps):
                feed_dict = {'inputs:0': self.test_inputs[step],
                    'labels:0': self.test_labels[step]}
                ret = sess.run(fetch, feed_dict)
                ret_losses = ret
                # sum of losses
                for i in range(len(self.losses)):
                    losses_sum[i] += ret_losses[i]
            # summary
            losses_mean = [l / self.epoch_steps for l in losses_sum]
            # stats
            ckpt_num = re.findall(prefix + r'(\d+)', ckpt)[0]
            stats.append(np.array([float(ckpt_num)] + losses_mean))
        # save stats
        import matplotlib.pyplot as plt
        stats = np.stack(stats)
        np.save(os.path.join(self.test_dir, 'stats.npy'), stats)
        # save plot
        fig, ax = plt.subplots()
        ax.set_title('Test Error with Training Progress')
        ax.set_xlabel('training steps')
        ax.set_ylabel('MAD (RGB)')
        ax.set_xscale('linear')
        ax.set_yscale('log')
        stats = stats[1:]
        ax.plot(stats[:, 0], stats[:, 2])
        plt.tight_layout()
        plt.savefig(os.path.join(self.test_dir, 'stats.png'))
        plt.close()

    def __call__(self):
        self.initialize()
        self.get_dataset()
        with tf.Graph().as_default():
            self.build_graph()
            self.build_saver()
            with create_session() as sess:
                self.run_last(sess)
                self.run_steps(sess)

def main(argv=None):
    # arguments parsing
    import argparse
    argp = argparse.ArgumentParser(argv[0])
    # testing parameters
    argp.add_argument('dataset')
    argp.add_argument('--num-epochs', type=int, default=1)
    argp.add_argument('--random-seed', type=int)
    argp.add_argument('--device', default='/gpu:0')
    argp.add_argument('--postfix', default='')
    argp.add_argument('--train-dir', default='./train{postfix}.tmp')
    argp.add_argument('--test-dir', default='./test{postfix}.tmp')
    argp.add_argument('--log-file', default='test.log')
    argp.add_argument('--batch-size', type=int)
    # data parameters
    argp.add_argument('--dtype', type=int, default=2)
    argp.add_argument('--data-format', default='NCHW')
    argp.add_argument('--in-channels', type=int, default=3)
    argp.add_argument('--out-channels', type=int, default=3)
    # pre-processing parameters
    Data.add_arguments(argp, True)
    # model parameters
    Model.add_arguments(argp)
    argp.add_argument('--scaling', type=int, default=1)
    # parse
    args = argp.parse_args(argv[1:])
    Data.parse_arguments(args)
    args.train_dir = args.train_dir.format(postfix=args.postfix)
    args.test_dir = args.test_dir.format(postfix=args.postfix)
    args.dtype = [tf.int8, tf.float16, tf.float32, tf.float64][args.dtype]
    args.pre_down = True
    # run testing
    test = Test(args)
    test()

if __name__ == '__main__':
    import sys
    main(sys.argv)
