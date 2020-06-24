import os
import random
import numpy as np
from scipy import ndimage
from PIL import Image
import zimg
from utils import eprint, reset_random, listdir_files, bool_argument

def random_resize(src, dw, dh, roi_left=0, roi_top=0, roi_width=0, roi_height=0, channel_first=False):
    rand0 = np.random.uniform(0, 1)
    if rand0 < 0.02:
        dst = zimg.resize(src, dw, dh, 'Point', channel_first=channel_first,
            roi_left=roi_left, roi_top=roi_top, roi_width=roi_width, roi_height=roi_height)
    elif rand0 < 0.04:
        dst = zimg.resize(src, dw, dh, 'Bilinear', channel_first=channel_first,
            roi_left=roi_left, roi_top=roi_top, roi_width=roi_width, roi_height=roi_height)
    elif rand0 < 0.06:
        dst = zimg.resize(src, dw, dh, 'Spline16', channel_first=channel_first,
            roi_left=roi_left, roi_top=roi_top, roi_width=roi_width, roi_height=roi_height)
    elif rand0 < 0.08:
        dst = zimg.resize(src, dw, dh, 'Spline36', channel_first=channel_first,
            roi_left=roi_left, roi_top=roi_top, roi_width=roi_width, roi_height=roi_height)
    elif rand0 < 0.10:
        dst = zimg.resize(src, dw, dh, 'Spline64', channel_first=channel_first,
            roi_left=roi_left, roi_top=roi_top, roi_width=roi_width, roi_height=roi_height)
    elif rand0 < 0.28: # Lanczos(taps=2~19)
        taps = np.random.randint(2, 20)
        dst = zimg.resize(src, dw, dh, 'Lanczos', taps, channel_first=channel_first,
            roi_left=roi_left, roi_top=roi_top, roi_width=roi_width, roi_height=roi_height)
    else: # Bicubic
        if rand0 < 0.46:
            if rand0 < 0.30: # Hermite
                B = 0
                C = 0
            elif rand0 < 0.32: # B-Spline
                B = 1
                C = 0
            elif rand0 < 0.34: # Robidoux Soft
                B = 0.67962275898295921 # (9-3*sqrt(2))/7
                C = 0.1601886205085204 # 0.5 - B * 0.5
            elif rand0 < 0.36: # Robidoux
                B = 0.37821575509399866 # 12/(19+9*sqrt(2)
                C = 0.31089212245300067 # 113/(58+216*sqrt(2))
            elif rand0 < 0.38: # Mitchell
                B = 1 / 3
                C = 1 / 3
            elif rand0 < 0.40: # Robidoux Sharp
                B = 0.2620145123990142 # 6/(13+7*sqrt(2))
                C = 0.3689927438004929 # 7/(2+12*sqrt(2)
            else: # Catmull-Rom
                B = 0
                C = 0.5
            # randomly alternate the kernel with 80% probability
            if np.random.randint(0, 10) > 1:
                B += np.random.normal(0, 0.05)
                C += np.random.normal(0, 0.05)
        elif rand0 < 0.80:
            if rand0 < 0.66: # Keys Cubic
                B = np.random.uniform(0, 2 / 3) + np.random.normal(0, 1 / 3)
                C = 0.5 - B * 0.5
            elif rand0 < 0.74: # Soft Cubic
                B = np.random.uniform(0.5, 1) + np.random.normal(0, 0.25)
                C = 1 - B
            else: # Sharp Cubic
                B = np.random.uniform(-0.75, -0.25) + np.random.normal(0, 0.25)
                C = B * -0.5
            # randomly alternate the kernel with 70%/90% probability
            if np.random.randint(0, 10) > (2 if rand0 < 0.66 else 0):
                B += np.random.normal(0, 1 / 6)
                C += np.random.normal(0, 1 / 6)
        elif rand0 < 0.85:
            B = np.random.uniform(-1.5, 1.5) # amount of haloing
            C = -1 # when c is around b * 0.8, aliasing is minimum
            if B >= 0: # with aliasing
                B = 1 + B
                while C < 0 or C > B * 1.2:
                    C = np.random.normal(B * 0.4, B * 0.2)
            else: # without aliasing
                B = 1 - B
                while C < 0 or C > B * 1.2:
                    C = np.random.normal(B * 0.8, B * 0.2)
            B = -B
            # randomly alternate the kernel
            B += np.random.normal(0, 0.25)
            C += np.random.normal(0, 0.25)
        else: # arbitrary Bicubic
            B = np.random.uniform(-2, 2) + np.random.normal(0, 0.5)
            C = np.random.uniform(-1, 2) + np.random.normal(0, 0.5)
        dst = zimg.resize(src, dw, dh, 'Bicubic', B, C, channel_first=channel_first,
            roi_left=roi_left, roi_top=roi_top, roi_width=roi_width, roi_height=roi_height)
    return dst

def random_filter(src, dw=None, dh=None, channel_first=False):
    last = src
    sw = src.shape[-1 if channel_first else -2]
    sh = src.shape[-2 if channel_first else -3]
    if dw is None:
        dw = sw
    if dh is None:
        dh = sh
    scale = np.sqrt((dw * dh) / (sw * sh))
    # random number
    rand_updown = np.random.randint(0, 5) # int ~ [0, 5)
    min_scale = min(-2, np.log2(scale * 0.5))
    max_scale = max(1, np.log2(scale * 2))
    rand_scale = np.random.uniform(min_scale, 0) if rand_updown > 0 else np.random.uniform(0, max_scale)
    rand_scale = 2 ** rand_scale # [0.25, 1) + [1, 4)
    # random resize
    tw = int(sw * rand_scale + 0.5)
    th = int(sh * rand_scale + 0.5)
    # print('{}x{} => {}x{} => {}x{}'.format(sw, sh, tw, th, dw, dh))
    last = random_resize(last, tw, th, channel_first=channel_first)
    last = random_resize(last, dw, dh, channel_first=channel_first)
    # return
    return last

def random_noise(src, noise_str=0.03, noise_corr=0.75, matrix=None, channel_first=False):
    last = src
    if matrix is None:
        matrix = ['BT709', 'ST170_M', 'BT2020_NCL'][np.random.randint(0, 3)]
    # noise generator
    def noise_gen(shape, scale=0.01, corr=0.0, channel_first=False):
        noise = np.random.normal(0.0, scale, shape).astype(np.float32)
        if corr > 0:
            corr = corr if len(shape) < 3 else [0, corr, corr] if channel_first else [corr, corr, 0]
            noise = ndimage.gaussian_filter(noise, corr, truncate=3.0)
        return noise
    # noise shape, scale and spatial correlation
    shapeRGB = last.shape
    shapeY = last.shape[1:] if channel_first else last.shape[:-1]
    corrY = np.abs(np.random.normal(0.0, noise_corr))
    corrY = 0 if corrY > noise_corr * 3 else corrY # no correlation if > sigma*3
    scaleY = np.abs(np.random.normal(0.0, noise_str)) * (1 + corrY)
    corrC = np.abs(np.random.normal(0.0, noise_corr))
    corrC = 0 if corrC > noise_corr * 3 else corrC # no correlation if > sigma*3
    scaleC = np.abs(np.random.normal(0.0, noise_str)) * (1 + corrC)
    # noise type
    rand_noise = np.random.randint(0, 10)
    # print('{}, Y: {}|{}, C: {}|{}'.format(rand_noise, scaleY, corrY, scaleC, corrC))
    if rand_noise < 2: # RGB noise
        noise = noise_gen(shapeRGB, scaleY, corrY, channel_first=channel_first)
        last = last + noise
    elif rand_noise < 4: # YUV444 noise
        noiseY = noise_gen(shapeY, scaleY, corrY, channel_first=channel_first)
        noiseU = noise_gen(shapeY, scaleC, corrC, channel_first=channel_first)
        noiseV = noise_gen(shapeY, scaleC, corrC, channel_first=channel_first)
        noise = np.stack([noiseY, noiseU, noiseV], axis=0 if channel_first else -1)
        noise = zimg.convertFormat(noise, channel_first=channel_first, matrix_in=matrix, matrix='rgb')
        last = last + noise
    elif rand_noise < 7: # Y noise
        noiseY = noise_gen(shapeY, scaleY, corrY, channel_first=channel_first)
        noise = np.stack([noiseY] * 3, axis=0 if channel_first else -1)
        last = last + noise
    # return
    return last

def random_chroma(src, matrix=None, channel_first=False):
    last = src
    sw = src.shape[-1 if channel_first else -2]
    sh = src.shape[-2 if channel_first else -3]
    if matrix is None:
        matrix = ['BT709', 'ST170_M', 'BT2020_NCL'][np.random.randint(0, 3)]
    # 0: YUV420, MPEG-1 chroma placement
    # 1: YUV420, MPEG-2 chroma placement
    # 2~4: RGB
    filters = (
        [{'filter': 'Bicubic', 'filter_a': 0, 'filter_b': 0.5}] * 3 +
        [{'filter': 'Bicubic', 'filter_a': 1/3, 'filter_b': 1/3}] * 2 +
        [{'filter': 'Bicubic', 'filter_a': 0.75, 'filter_b': 0.25},
        {'filter': 'Bicubic', 'filter_a': 1.0, 'filter_b': 0.0},
        {'filter': 'Point'}, {'filter': 'Bilinear'},
        {'filter': 'Lanczos', 'filter_a': 3}]
    )
    rand_yuv = np.random.randint(0, 5)
    # convert RGB to YUV420
    if rand_yuv < 2:
        last = zimg.convertFormat(last, channel_first=channel_first, matrix_in='rgb', matrix=matrix)
        lastY = last[0] if channel_first else last[:, :, 0]
        lastU = last[1] if channel_first else last[:, :, 1]
        lastV = last[2] if channel_first else last[:, :, 2]
        filter_params = filters[np.random.randint(0, len(filters))]
        resizer = zimg.Resizer.createScale(lastU, 0.5, **filter_params, channel_first=channel_first,
            roi_left=0 if rand_yuv % 2 == 0 else -0.5)
        lastU = resizer(lastU)
        lastV = resizer(lastV)
    # convert YUV420 to RGB
    if rand_yuv < 2:
        filter_params = filters[np.random.randint(0, len(filters))]
        resizer = zimg.Resizer.create(lastU, sw, sh, **filter_params, channel_first=channel_first,
            roi_left=0 if rand_yuv % 2 == 0 else 0.25)
        lastU = resizer(lastU)
        lastV = resizer(lastV)
        last = np.stack((lastY, lastU, lastV), axis=0 if channel_first else -1)
        last = zimg.convertFormat(last, channel_first=channel_first, matrix_in=matrix, matrix='rgb')
    # return
    return last

def linear_resize(src, dw, dh, channel_first=False):
    last = src
    # convert to linear scale
    last = zimg.convertFormat(last, channel_first=channel_first, transfer_in='bt709', transfer='linear')
    # resize
    last = zimg.resize(last, dw, dh, 'Bicubic', 0, 0.5, channel_first=channel_first)
    # convert back to gamma-corrected scale
    last = zimg.convertFormat(last, channel_first=channel_first, transfer_in='linear', transfer='bt709')
    # return
    return last

def convert_dtype(img, dtype):
    if dtype == img.dtype:
        pass
    elif dtype == np.float32:
        if img.dtype == np.uint8:
            img = np.float32(img) * (1 / 255)
        elif img.dtype == np.uint16:
            img = np.float32(img) * (1 / 65535)
        elif img.dtype != np.float32:
            img = np.float32(img)
    elif dtype == np.uint16:
        if img.dtype == np.uint8:
            img = np.uint16(img) * 255
        elif img.dtype != np.uint16:
            img = np.clip(img, 0, 1)
            img = np.uint16(img * 65535 + 0.5)
    elif dtype == np.uint8:
        if img.dtype == np.uint16:
            img = np.uint8(img // 257)
        elif img.dtype != np.uint8:
            img = np.clip(img, 0, 1)
            img = np.uint8(img * 255 + 0.5)
    return img

def pre_process(config, img, dtype=np.float32):
    channel_first = True
    # image dimension regularization
    rank = len(img.shape)
    if rank == 2:
        img = np.stack([img] * 3, axis=-3) # HW => CHW
    elif rank == 3:
        img = np.transpose(img, (2, 0, 1)) # HWC => CHW
    channels = img.shape[-3]
    if channels < 3: # Gray
        if channels == 2: # with alpha
            img = img[0:1]
        img = np.concatenate([img] * 3, axis=-3)
    elif channels == 4: # RGB with alpha
        img = img[0:3]
    height = img.shape[-2]
    width = img.shape[-1]
    # pre downscale ratio for high-resolution image
    pre_scale = 1
    if config.pre_down:
        if (width >= 3072 and height >= 1536) or (width >= 1536 and height >= 3072):
            pre_scale = 3
        elif (width >= 1536 and height >= 768) or (width >= 768 and height >= 1536):
            pre_scale = 2
    # cropping
    cropped_height = config.patch_height * pre_scale
    cropped_width = config.patch_width * pre_scale
    offset_height = np.random.randint(0, height - cropped_height + 1) if height > cropped_height else 0
    offset_width = np.random.randint(0, width - cropped_width + 1) if width > cropped_width else 0
    img = img[:, offset_height : offset_height + cropped_height, offset_width : offset_width + cropped_width]
    height = min(height, cropped_height)
    width = min(width, cropped_width)
    # padding
    if width < cropped_width or height < cropped_height:
        pad_height = max(0, cropped_height - height)
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_width = max(0, cropped_width - width)
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        img = np.pad(img, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
        height = max(height, cropped_height)
        width = max(width, cropped_width)
    # random transpose with 50% probability
    if np.random.randint(0, 2) > 0:
        img = np.transpose(img, (0, 2, 1))
    # random flipping with 25% probability each
    rand_val = np.random.randint(0, 4)
    if rand_val == 1:
        img = img[:, :, ::-1]
    if rand_val == 2:
        img = img[:, ::-1, :]
    if rand_val == 3:
        img = img[:, ::-1, ::-1]
    # convert to float32
    img2 = convert_dtype(img, np.float32)
    # random filter for input
    rand_linear = np.random.randint(0, 2) # int ~ [0, 2)
    matrix = ['BT709', 'ST170_M', 'BT2020_NCL'][np.random.randint(0, 3)]
    _input = img2
    if rand_linear > 0: # randomly convert to linear scale
        _input = zimg.convertFormat(_input, channel_first=channel_first, transfer_in=config.transfer, transfer='LINEAR')
    _input = random_filter(_input, config.patch_width // config.scale, config.patch_height // config.scale,
        channel_first=channel_first) # random filtering with resizer
    if config.noise_str > 0: # random noise
        _input = random_noise(_input, config.noise_str, config.noise_corr, matrix=matrix, channel_first=channel_first)
    if rand_linear > 0: # convert back to gamma-corrected scale
        _input = zimg.convertFormat(_input, channel_first=channel_first, transfer_in='LINEAR', transfer=config.transfer)
    _input = random_chroma(_input, matrix=matrix, channel_first=channel_first)
    # type conversion (input)
    _input = convert_dtype(_input, dtype)
    # pre downscale and type conversion (label)
    if pre_scale != 1:
        _label = linear_resize(img2, config.patch_width, config.patch_height, channel_first=channel_first)
        _label = convert_dtype(_label, dtype)
    elif dtype == np.float32:
        _label = img2
    else:
        _label = convert_dtype(img, dtype)
    # return
    return _input, _label # CHW, dtype

class DataWriter:
    def __init__(self, config):
        self.config = config

    @classmethod
    def initialize(cls, config):
        # create save directory
        if os.path.exists(config.save_dir):
            eprint('Confirm removing {}\n[Y/n]'.format(config.save_dir))
            if input() == 'Y':
                import shutil
                shutil.rmtree(config.save_dir)
                eprint('Removed: ' + config.save_dir)
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)
        # set deterministic random seed
        if config.random_seed is not None:
            reset_random(config.random_seed)

    @classmethod
    def get_dataset(cls, config):
        exts = ['.bmp', '.png', '.jpg', '.jpeg', '.webp', '.jp2', '.tiff']
        dataset = listdir_files(config.input_dir, recursive=True, filter_ext=exts)
        return dataset

    @staticmethod
    def process(config, ifile, ofile):
        im = Image.open(ifile)
        img = np.array(im, copy=False)
        _input, _label = pre_process(config, img, np.dtype(config.dtype))
        np.savez_compressed(ofile, input=_input, label=_label)

    @classmethod
    def run(cls, config, dataset):
        _dataset = dataset.copy()
        epochs = config.epochs
        epoch_steps = len(_dataset)
        # pre-shuffle the dataset
        if config.shuffle == 1:
            random.shuffle(_dataset)
        # execute pre-process
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(config.processes) as executor:
            futures = []
            for epoch in range(epochs):
                # create directory for each epoch
                odir = os.path.join(config.save_dir, '{:0>{width}}'.format(epoch, width=len(str(epochs))))
                if not os.path.exists(odir):
                    os.makedirs(odir)
                # randomly shuffle for each epoch
                if config.shuffle == 2:
                    random.shuffle(_dataset)
                # loop over the steps and append the calls
                for step in range(epoch_steps):
                    ifile = _dataset[step]
                    ofile = os.path.join(odir, '{:0>{width}}'.format(step, width=len(str(epoch_steps))))
                    # skip existing files
                    if not os.path.exists(ofile):
                        futures.append(executor.submit(cls.process, config, ifile, ofile))
                # execute the calls
                for future in futures:
                    future.result()

    def __call__(self):
        self.initialize(self.config)
        dataset = self.get_dataset(self.config)
        self.run(self.config, dataset)

def main(argv):
    import argparse
    argp = argparse.ArgumentParser(argv[0])
    argp.add_argument('input_dir')
    argp.add_argument('save_dir')
    argp.add_argument('--random-seed', type=int)
    argp.add_argument('--epochs', type=int, default=1)
    argp.add_argument('--shuffle', type=int, default=1) # 0: no shuffle, 1: shuffle once, 2: shuffle every epoch
    argp.add_argument('--processes', type=int, default=8)
    argp.add_argument('--dtype', default='uint8')
    argp.add_argument('--pre-down', type=bool, default=1)
    argp.add_argument('--scale', type=int, default=1)
    argp.add_argument('--patch-width', type=int, default=256)
    argp.add_argument('--patch-height', type=int, default=256)
    argp.add_argument('--transfer', default='BT709')
    argp.add_argument('--noise-str', type=float, default=0.03)
    argp.add_argument('--noise-corr', type=float, default=0.75)
    # parse
    args = argp.parse_args(argv[1:])
    # run data writer
    writer = DataWriter(args)
    writer()

if __name__ == '__main__':
    import sys
    main(sys.argv)
