import tensorflow as tf
import numpy as np
import random
from utils import eprint, listdir_files

class Data:
    def __init__(self, config):
        self.dataset = None
        self.num_epochs = None
        self.max_steps = None
        self.batch_size = None
        self.val_size = None
        self.threads = None
        self.prefetch = None
        self.buffer_size = None
        # copy all the properties from config object
        self.config = config
        self.__dict__.update(config.__dict__)
        # initialize
        self.get_files()

    @staticmethod
    def add_arguments(argp):
        # pre-processing parameters
        argp.add_argument('--threads', type=int, default=4)
        argp.add_argument('--prefetch', type=int, default=64)
        argp.add_argument('--buffer-size', type=int, default=1024)

    def get_files(self):
        data_list = listdir_files(self.dataset, filter_ext=['.npz'])
        # val set
        if self.val_size is not None:
            assert self.val_size < len(data_list)
            self.val_steps = self.val_size // self.batch_size
            self.val_size = self.val_steps * self.batch_size
            self.val_set = data_list[:self.val_size]
            data_list = data_list[self.val_size:]
            eprint('validation set: {}'.format(self.val_size))
        # main set
        assert self.batch_size <= len(data_list)
        self.epoch_steps = len(data_list) // self.batch_size
        self.epoch_size = self.epoch_steps * self.batch_size
        if self.max_steps is None:
            self.max_steps = self.epoch_steps * self.num_epochs
        else:
            self.num_epochs = (self.max_steps + self.epoch_steps - 1) // self.epoch_steps
        self.main_set = data_list[:self.epoch_size]
        # print
        eprint('main set: {}\nepoch steps: {}\nnum epochs: {}\nmax steps: {}\n'
            .format(len(self.main_set), self.epoch_steps, self.num_epochs, self.max_steps))

    @staticmethod
    def process_sample(file):
        with np.load(file) as loaded:
            return loaded['inputs'], loaded['labels']

    def extract_batch(self, batch_set):
        # initialize
        inputs = []
        labels = []
        # load all the data
        for file in batch_set:
            _input, _label = Data.process_sample(file)
            inputs.append(_input)
            labels.append(_label)
        # stack data to form a batch
        inputs = np.stack(inputs)
        labels = np.stack(labels)
        return inputs, labels

    def _gen_batches(self, dataset, epoch_steps, num_epochs=1, start=0,
        shuffle=False):
        _dataset = dataset
        max_steps = epoch_steps * num_epochs
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(self.threads) as executor:
            futures = []
            # loop over epochs
            for epoch in range(start // epoch_steps, num_epochs):
                step_offset = epoch_steps * epoch
                step_start = max(0, start - step_offset)
                step_stop = min(epoch_steps, max_steps - step_offset)
                # random shuffle
                if shuffle:
                    _dataset = dataset.copy()
                    random.shuffle(_dataset)
                # loop over steps within an epoch
                for step in range(step_start, step_stop):
                    offset = step * self.batch_size
                    upper = min(len(_dataset), offset + self.batch_size)
                    batch_set = _dataset[offset : upper]
                    futures.append(executor.submit(self.extract_batch, batch_set))
                    # yield the data beyond prefetch range
                    while len(futures) >= self.prefetch:
                        future = futures.pop(0)
                        yield future.result()
            # yield the remaining data
            for future in futures:
                yield future.result()

    def gen_main(self, start=0):
        return self._gen_batches(self.main_set, self.epoch_steps, self.num_epochs,
            start, True)

    def gen_val(self, start=0):
        return self._gen_batches(self.val_set, self.val_steps, 1,
            start, False)
