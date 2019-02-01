import tensorflow as tf
import numpy as np

def bool_argument(argp, name, default):
    argp.add_argument('--' + name, dest=name, action='store_true')
    argp.add_argument('--no-' + name, dest=name, action='store_false')
    eval('argp.set_defaults({}={})'.format(name, 'True' if default else 'False'))

# stderr print
def eprint(*args, **kwargs):
    import sys
    print(*args, file=sys.stderr, **kwargs)

# get dirname of file, supporting relative path
def get_dirname(file):
    import os
    return os.path.dirname(os.path.abspath(file))

# make directories recursively
def makedirs(dirname):
    import os
    import errno
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

def makedirs_f(file):
    makedirs(get_dirname(file))

# create parent directories if not exist
def open_auto(file, *args, **kwargs):
    makedirs_f(file)
    return open(file, *args, **kwargs)

# get the line number of specific file
def file_lines(file, encoding='utf-8', *args, **kwargs):
    with open(file, encoding=encoding, *args, **kwargs) as fd:
        for i, _ in enumerate(fd):
            pass
    return i + 1

# read csv into pandas with Unicode path
def read_csv(file, encoding='utf-8', *args, **kwargs):
    import pandas as pd
    with open(file, 'r', encoding=encoding) as fd:
        return pd.read_csv(fd, *args, **kwargs)

# create parent directories if not exist
def to_csv_auto(dataframe, file, *args, **kwargs):
    makedirs_f(file)
    with open(file, 'w', encoding='utf-8') as fd:
        dataframe.to_csv(fd, *args, **kwargs)

# append to an existing csv file
def append_to_csv(dataframe, file, *args, **kwargs):
    import os
    if os.path.exists(file):
        with open(file, 'a', encoding='utf-8') as fd:
            dataframe.to_csv(fd, header=False, *args, **kwargs)
    else:
        to_csv_auto(dataframe, file, *args, **kwargs)

# recursively list all the files' path under directory
def listdir_files(path, recursive=True, filter_ext=None, encoding=None):
    import os, locale
    if encoding is True: encoding = locale.getpreferredencoding()
    if filter_ext is not None: filter_ext = [e.lower() for e in filter_ext]
    files = []
    for (dir_path, dir_names, file_names) in os.walk(path):
        for f in file_names:
            if not filter_ext or os.path.splitext(f)[1].lower() in filter_ext:
                file_path = os.path.join(dir_path, f)
                try:
                    if encoding: file_path = file_path.encode(encoding)
                    files.append(file_path)
                except UnicodeEncodeError as err:
                    eprint(file_path)
                    eprint(err)
        if not recursive: break
    return files

# reset random seeds
def reset_random(seed=0):
    import random
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

# setup tensorflow and return session
def create_session(graph=None, debug=False):
    # create session
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options,
        allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(graph=graph, config=config)
    if debug:
        from tensorflow.python import debug as tfdbg
        sess = tfdbg.LocalCLIDebugWrapperSession(sess)
    return sess

# encode a batch of images to a list of pngs
def BatchPNG(images, batch_size, dtype=tf.uint8):
    pngs = []
    for i in range(batch_size):
        img = images[i]
        if img.dtype != tf.uint8: img = tf.image.convert_image_dtype(img, dtype, saturate=True)
        png = tf.image.encode_png(img, compression=9)
        pngs.append(png)
    return pngs
