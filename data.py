from abc import ABCMeta, abstractmethod
import numpy as np
import os
import random
from utils import bool_argument, eprint, listdir_files

def convert_dtype(img, dtype):
    src_dtype = img.dtype
    if dtype == src_dtype: # skip same type
        return img
    elif dtype == np.uint16:
        if src_dtype == np.uint8:
            img = np.uint16(img) * 257
        elif src_dtype != np.uint16:
            img = np.clip(img, 0, 1)
            img = np.uint16(img * 65535 + 0.5)
    elif dtype == np.uint8:
        if src_dtype == np.uint16:
            img = np.uint8((np.int32(img) + 128) // 257)
        elif src_dtype != np.uint8:
            img = np.clip(img, 0, 1)
            img = np.uint8(img * 255 + 0.5)
    else: # assume float
        img = img.astype(dtype)
        if src_dtype == np.uint8:
            img *= (1 / 255)
        elif src_dtype == np.uint16:
            img *= (1 / 65535)
    # return
    return img

# ======
# base class

class DataBase:
    __metaclass__ = ABCMeta

    def __init__(self, config):
        self.dataset = None
        self.val_dir = None
        self.num_epochs = None
        self.max_steps = None
        self.batch_size = None
        self.val_size = None
        self.packed = None
        self.processes = None
        self.threads = None
        self.prefetch = None
        self.buffer_size = None
        self.shuffle = None
        self.mixup = None
        # copy all the properties from config object
        self.config = config
        self.__dict__.update(config.__dict__)
        # initialize
        self.val_set = None
        self.get_files()

    @staticmethod
    def add_arguments(argp, test=False):
        # base parameters
        bool_argument(argp, 'packed', False)
        bool_argument(argp, 'test', test)
        # pre-processing parameters
        argp.add_argument('--processes', type=int, default=4)
        argp.add_argument('--threads', type=int, default=1)
        argp.add_argument('--prefetch', type=int, default=64)
        argp.add_argument('--buffer-size', type=int, default=256)
        bool_argument(argp, 'shuffle', True)
        bool_argument(argp, 'mixup', False)

    @staticmethod
    def parse_arguments(args):
        def argdefault(name, value):
            if args.__getattribute__(name) is None:
                args.__setattr__(name, value)
        def argchoose(name, cond, tv, fv):
            argdefault(name, tv if cond else fv)
        argchoose('batch_size', args.test, 1, 32)
        # force packed data loader if enable mixup
        if args.mixup:
            args.packed = True

    def get_files_packed(self):
        data_list = listdir_files(self.dataset, recursive=True, filter_ext=['.npz'])
        if self.shuffle:
            random.shuffle(data_list)
        # val set
        if self.val_dir is not None:
            val_set = listdir_files(self.val_dir, recursive=True, filter_ext=['.npz'])
            self.val_steps = len(val_set)
            self.val_size = self.val_steps * self.batch_size
            self.val_set = val_set[:self.val_steps]
            eprint('validation set: {}'.format(self.val_size))
        elif self.val_size is not None:
            self.val_steps = self.val_size // self.batch_size
            assert self.val_steps < len(data_list)
            self.val_size = self.val_steps * self.batch_size
            self.val_set = data_list[:self.val_steps]
            data_list = data_list[self.val_steps:]
            eprint('validation set: {}'.format(self.val_size))
        # main set
        self.epoch_steps = len(data_list)
        self.epoch_size = self.epoch_steps * self.batch_size
        if self.max_steps is None:
            self.max_steps = self.epoch_steps * self.num_epochs
        else:
            self.num_epochs = (self.max_steps + self.epoch_steps - 1) // self.epoch_steps
        self.main_set = data_list

    @abstractmethod
    def get_files_origin(self):
        pass

    def get_files(self):
        if self.packed: # packed dataset
            self.get_files_packed()
        else: # non-packed dataset
            data_list = self.get_files_origin()
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
        # write val set to file
        if self.val_set is not None and self.config.__contains__('train_dir'):
            with open(os.path.join(self.config.train_dir, 'val_set.txt'), 'w') as fd:
                fd.writelines(['{}\n'.format(i) for i in self.val_set])
        # print
        eprint('main set: {}\nepoch steps: {}\nnum epochs: {}\nmax steps: {}\n'
            .format(self.epoch_size, self.epoch_steps, self.num_epochs, self.max_steps))

    @staticmethod
    def process_sample(file, label, config):
        pass

    @classmethod
    def extract_batch(cls, batch_set, config):
        # initialize
        inputs = []
        labels = []
        # load all data in the batch
        for file in batch_set:
            with np.load(file) as npz:
                _input = npz['inputs']
                _label = npz['labels']
            inputs.append(_input)
            labels.append(_label)
        # concat data to form a batch (NCHW)
        if len(inputs[0].shape) < 4:
            inputs = np.stack(inputs, axis=0)
        else:
            inputs = np.concatenate(inputs, axis=0)
        if len(labels[0].shape) < 4:
            labels = np.stack(labels, axis=0)
        else:
            labels = np.concatenate(labels, axis=0)
        # convert to float32
        inputs = convert_dtype(inputs, np.float32)
        labels = convert_dtype(labels, np.float32)
        # return
        return inputs, labels

    @classmethod
    def extract_batch_packed(cls, batch_set):
        # load the batch
        with np.load(batch_set) as npz:
            inputs = npz['inputs']
            labels = npz['labels']
        # convert to float32
        inputs = convert_dtype(inputs, np.float32)
        labels = convert_dtype(labels, np.float32)
        # return
        return inputs, labels

    @classmethod
    def linear2gamma(cls, last, epsilon=1e-8):
        power = 1 / 2.4
        slope = 12.9232102
        alpha = 1.055
        k0 = 11 / 280
        beta = k0 / slope
        last = np.where(last < beta, slope * last,
            alpha * (last + epsilon) ** power - (alpha - 1))
        return last

    @classmethod
    def extract_batch_mixup(cls, batch_set, batch_set2):
        # load the batch
        with np.load(batch_set) as npz:
            inputs = npz['inputs']
            labels = npz['labels']
        with np.load(batch_set2) as npz:
            inputs2 = npz['inputs']
            labels2 = npz['labels']
        # convert to float32
        inputs = convert_dtype(inputs, np.float32)
        labels = convert_dtype(labels, np.float32)
        inputs2 = convert_dtype(inputs2, np.float32)
        labels2 = convert_dtype(labels2, np.float32)
        # linear to gamma
        inputs = cls.linear2gamma(inputs)
        inputs2 = cls.linear2gamma(inputs2)
        labels = cls.linear2gamma(labels)
        labels2 = cls.linear2gamma(labels2)
        # mixup
        alpha = 1.2
        _lambda = np.random.beta(alpha, alpha, (inputs.shape[0], 1, 1, 1))
        inputs = _lambda * inputs + (1 - _lambda) * inputs2
        labels = _lambda * labels + (1 - _lambda) * labels2
        # return
        return inputs, labels

    def _gen_batches_packed(self, dataset, epoch_steps, num_epochs=1, start=0,
        shuffle=False):
        _dataset = dataset.copy()
        _dataset2 = dataset.copy() # mixup dataset
        max_steps = epoch_steps * num_epochs
        # multi-process
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(self.processes) as executor:
            futures = []
            # loop over epochs
            for epoch in range(start // epoch_steps, num_epochs):
                step_offset = epoch_steps * epoch
                step_start = max(0, start - step_offset)
                step_stop = min(epoch_steps, max_steps - step_offset)
                # random shuffle
                if shuffle:
                    random.shuffle(_dataset)
                if self.mixup: # force shuffle for mixup dataset
                    random.shuffle(_dataset2)
                # loop over steps within an epoch
                for step in range(step_start, step_stop):
                    batch_set = _dataset[step]
                    if self.mixup:
                        batch_set2 = _dataset2[step]
                        futures.append(executor.submit(self.extract_batch_mixup, batch_set, batch_set2))
                    else:
                        futures.append(executor.submit(self.extract_batch_packed, batch_set))
                    # yield the data beyond prefetch range
                    while len(futures) >= self.prefetch:
                        yield futures.pop(0).result()
            # yield the remaining data
            for future in futures:
                yield future.result()

    def _gen_batches_origin(self, dataset, epoch_steps, num_epochs=1, start=0,
        shuffle=False):
        _dataset = dataset.copy()
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
                    random.shuffle(_dataset)
                # loop over steps within an epoch
                for step in range(step_start, step_stop):
                    offset = step * self.batch_size
                    upper = min(len(_dataset), offset + self.batch_size)
                    batch_set = _dataset[offset : upper]
                    futures.append(executor.submit(self.extract_batch,
                        batch_set, self.config))
                    # yield the data beyond prefetch range
                    while len(futures) >= self.prefetch:
                        yield futures.pop(0).result()
            # yield the remaining data
            for future in futures:
                yield future.result()

    def _gen_batches(self, dataset, epoch_steps, num_epochs=1, start=0,
        shuffle=False):
        # packed dataset
        if self.packed:
            return self._gen_batches_packed(dataset, epoch_steps, num_epochs, start, shuffle)
        else:
            return self._gen_batches_origin(dataset, epoch_steps, num_epochs, start, shuffle)

    def gen_main(self, start=0):
        return self._gen_batches(self.main_set, self.epoch_steps, self.num_epochs,
            start, self.shuffle)

    def gen_val(self, start=0):
        return self._gen_batches(self.val_set, self.val_steps, 1,
            start, False)

# ======
# derived classes

class DataImage(DataBase):
    def get_files_origin(self):
        data_list = listdir_files(self.dataset, recursive=True, filter_ext=['.npz'])
        # return
        if self.shuffle:
            random.shuffle(data_list)
        return data_list
