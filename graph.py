import tensorflow as tf
import os
from utils import eprint, create_session
from model import Model

class Graph:
    def __init__(self, config):
        self.postfix = None
        self.train_dir = None
        self.model_dir = None
        self.model_file = None
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
        if os.path.exists(self.model_dir):
            eprint('Confirm removing {}\n[Y/n]'.format(self.model_dir))
            if input() != 'Y':
                import sys
                sys.exit()
            import shutil
            shutil.rmtree(self.model_dir, ignore_errors=True)
            eprint('Removed: ' + self.model_dir)
        os.makedirs(self.model_dir)

    def build_graph(self):
        self.model = Model(self.config)
        self.model.build_model()

    def build_saver(self):
        # a Saver object to restore the variables with mappings
        self.saver_r = tf.train.Saver(self.model.rvars)
        # a Saver object to save the variables without mappings
        self.saver_s = tf.train.Saver(self.model.svars)

    def run(self, sess):
        # save the GraphDef
        tf.train.write_graph(tf.get_default_graph(), self.model_dir,
            'model.graphdef', as_text=True)
        # latest checkpoint or specific model
        if self.model_file is None:
            ckpt = tf.train.latest_checkpoint(self.train_dir)
        else:
            ckpt = os.path.join(self.train_dir, self.model_file)
        eprint('Loading model: {}'.format(ckpt))
        self.saver_r.restore(sess, ckpt)
        # save the model parameters
        self.saver_s.export_meta_graph(os.path.join(self.model_dir, 'model.meta'),
            as_text=False, clear_devices=True, clear_extraneous_savers=True)
        self.saver_s.save(sess, os.path.join(self.model_dir, 'model'),
            write_meta_graph=False, write_state=False)

    def __call__(self):
        self.initialize()
        with tf.Graph().as_default():
            self.build_graph()
            self.build_saver()
            with create_session() as sess:
                self.run(sess)

def main(argv=None):
    # arguments parsing
    import argparse
    argp = argparse.ArgumentParser()
    # testing parameters
    argp.add_argument('--postfix', default='')
    argp.add_argument('--train-dir', default='./train{postfix}.tmp')
    argp.add_argument('--model-dir', default='./model{postfix}.tmp')
    argp.add_argument('--model-file')
    # data parameters
    argp.add_argument('--dtype', type=int, default=2)
    argp.add_argument('--data-format', default='NCHW')
    argp.add_argument('--in-channels', type=int, default=3)
    argp.add_argument('--out-channels', type=int, default=3)
    # model parameters
    Model.add_arguments(argp)
    argp.add_argument('--scaling', type=int, default=1)
    # parse
    args = argp.parse_args(argv)
    args.train_dir = args.train_dir.format(postfix=args.postfix)
    args.model_dir = args.model_dir.format(postfix=args.postfix)
    args.dtype = [tf.int8, tf.float16, tf.float32, tf.float64][args.dtype]
    # save model
    graph = Graph(args)
    graph()

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
