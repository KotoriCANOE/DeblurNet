import tensorflow as tf
import numpy as np
from utils import eprint, listdir_files

class Data:
    def __init__(self, config):
        self.dataset = None
        self.batch_size = None
        self.num_epochs = None
        self.max_steps = None
        self.val_size = None
        # copy all the properties from config object
        self.config = config
        self.__dict__.update(config.__dict__)
        self.get_files()
    
    def get_files(self):
        files = listdir_files(self.dataset, filter_ext=['.npz'])
        # validation set
        if self.val_size is not None:
            self.val_set = files[:self.val_size]
            files = files[self.val_size:]
        # main set
        self.epoch_steps = len(files) // self.batch_size
        self.epoch_size = self.epoch_steps * self.batch_size
        if self.max_steps is None:
            self.max_steps = self.epoch_steps * self.num_epochs
        else:
            self.num_epochs = (self.max_steps + self.epoch_steps - 1) // self.epoch_steps
        self.main_set = files[:self.epoch_size]
        # print
        eprint('main set: {}\nepoch steps: {}\nnum epochs: {}\nmax steps: {}\n'
            .format(len(self.main_set), self.epoch_steps, self.num_epochs, self.max_steps))

    @staticmethod
    def extract_batch(dataset, offset, batch_size):
        inputs = []
        labels = []
        for f in dataset[offset : offset + batch_size]:
            with np.load(f) as loaded:
                inputs.append(loaded['inputs'])
                labels.append(loaded['labels'])
        inputs = np.stack(inputs)
        labels = np.stack(labels)
        return inputs, labels
    
    def _gen_main_sub(self, datalist, start=0):
        pass

    def gen_main(self, start=0):
        for epoch in range(start // self.epoch_steps, self.num_epochs):
            step_offset = self.epoch_steps * epoch
            step_start = max(0, start - step_offset)
            step_stop = min(self.epoch_steps, self.max_steps - step_offset)
            for step in range(step_start, step_stop):
                yield self.extract_batch(self.main_set,
                    step * self.batch_size, self.batch_size)

    def get_val(self):
        return self.extract_batch(self.val_set, 0, self.val_size)

def data_arguments(argp):
    # pre-processing parameters
    argp.add_argument('--threads-py', type=int, default=16)
    argp.add_argument('--buffer-size', type=int, default=0)

def get_data(config, files):
    # convert dtype
    def convert_dtype(arr):
        dtype = arr.dtype
        if dtype != np.float32:
            arr = arr.astype(np.float32)
        if dtype == np.uint8:
            arr *= 1 / 255
        elif dtype == np.uint16:
            arr *= 1 / 65535
        return arr
    # python parse function
    def parse_pyfunc(file):
        with np.load(file.decode()) as loaded:
            inputs = loaded['inputs']
            labels = loaded['labels']
        inputs = convert_dtype(config)
        labels = convert_dtype(config)
        return inputs, labels
    # Dataset API
    dataset = tf.data.Dataset.from_tensor_slices((files,))
    if config.buffer_size > 0: dataset = dataset.shuffle(config.buffer_size)
    dataset = dataset.map(lambda file: tuple(tf.py_func(parse_pyfunc,
                              [file], [tf.float32, tf.float32])),
                          num_parallel_calls=config.threads_py)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.repeat(config.num_epochs)
    dataset = dataset.prefetch(64)
    # return iterator
    iterator = dataset.make_one_shot_iterator()
    next_inputs, next_labels = iterator.get_next()
    # data shape declaration
    data_shape = [None] * 4
    data_shape[-3 if config.data_format == 'NCHW' else -1] = config.in_channels
    next_inputs.set_shape(data_shape)
    next_labels.set_shape(data_shape)
    # return
    return next_inputs, next_labels
