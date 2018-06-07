import tensorflow as tf
import numpy as np
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
        files = listdir_files(self.dataset, filter_ext=['.npz'])
        # val set
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
    def extract_batch(files):
        inputs = []
        labels = []
        # load all the data files
        for f in files:
            with np.load(f) as loaded:
                inputs.append(loaded['inputs'])
                labels.append(loaded['labels'])
        # stack data to form a batch
        inputs = np.stack(inputs)
        labels = np.stack(labels)
        return inputs, labels

    def gen_main(self, start=0):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(self.threads) as executor:
            futures = []
            # loop over epochs
            for epoch in range(start // self.epoch_steps, self.num_epochs):
                step_offset = self.epoch_steps * epoch
                step_start = max(0, start - step_offset)
                step_stop = min(self.epoch_steps, self.max_steps - step_offset)
                # loop over steps within an epoch
                for step in range(step_start, step_stop):
                    offset = step * self.batch_size
                    files = self.main_set[offset : offset + self.batch_size]
                    futures.append(executor.submit(self.extract_batch, files))
                    # yield the data beyond prefetch range
                    while len(futures) >= self.prefetch:
                        future = futures.pop(0)
                        yield future.result()
            # yield the remaining data
            for future in futures:
                yield future.result()

    def get_val(self):
        return self.extract_batch(self.val_set)
