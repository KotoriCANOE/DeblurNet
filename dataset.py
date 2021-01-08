import os
import random
import numpy as np
from scipy import ndimage
from PIL import Image
from io import BytesIO
import webp
import zimg
from time import time
from utils import eprint, reset_random, listdir_files, bool_argument

# NOTE: ZIMG implement BT709, BT601, BT2020 transfer as a gamma=2.4 curve, which differs from the standards

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

def random_resize(param, src, dw, dh, roi_left=0, roi_top=0, roi_width=0, roi_height=0, channel_first=False):
    rand_val = np.random.randint(0, 100)
    if rand_val < param['Point']:
        dst = zimg.resize(src, dw, dh, 'Point', channel_first=channel_first,
            roi_left=roi_left, roi_top=roi_top, roi_width=roi_width, roi_height=roi_height)
    elif rand_val < param['Bilinear']:
        dst = zimg.resize(src, dw, dh, 'Bilinear', channel_first=channel_first,
            roi_left=roi_left, roi_top=roi_top, roi_width=roi_width, roi_height=roi_height)
    elif rand_val < param['Spline16']:
        dst = zimg.resize(src, dw, dh, 'Spline16', channel_first=channel_first,
            roi_left=roi_left, roi_top=roi_top, roi_width=roi_width, roi_height=roi_height)
    elif rand_val < param['Spline36']:
        dst = zimg.resize(src, dw, dh, 'Spline36', channel_first=channel_first,
            roi_left=roi_left, roi_top=roi_top, roi_width=roi_width, roi_height=roi_height)
    elif rand_val < param['Spline64']:
        dst = zimg.resize(src, dw, dh, 'Spline64', channel_first=channel_first,
            roi_left=roi_left, roi_top=roi_top, roi_width=roi_width, roi_height=roi_height)
    elif rand_val < param['Lanczos']: # Lanczos(taps=2~19)
        taps = np.random.randint(2, 20)
        dst = zimg.resize(src, dw, dh, 'Lanczos', taps, channel_first=channel_first,
            roi_left=roi_left, roi_top=roi_top, roi_width=roi_width, roi_height=roi_height)
    else: # Bicubic
        if rand_val < param['Catmull-Rom']:
            if rand_val < param['Hermite']: # Hermite
                B = 0
                C = 0
            elif rand_val < param['B-Spline']: # B-Spline
                B = 1
                C = 0
            elif rand_val < param['RobidouxSoft']: # Robidoux Soft
                B = 0.67962275898295921 # (9-3*sqrt(2))/7
                C = 0.1601886205085204 # 0.5 - B * 0.5
            elif rand_val < param['Robidoux']: # Robidoux
                B = 0.37821575509399866 # 12/(19+9*sqrt(2)
                C = 0.31089212245300067 # 113/(58+216*sqrt(2))
            elif rand_val < param['Mitchell']: # Mitchell
                B = 1 / 3
                C = 1 / 3
            elif rand_val < param['RobidouxSharp']: # Robidoux Sharp
                B = 0.2620145123990142 # 6/(13+7*sqrt(2))
                C = 0.3689927438004929 # 7/(2+12*sqrt(2)
            else: # Catmull-Rom
                B = 0
                C = 0.5
            # randomly alternate the kernel with 80% probability
            if np.random.randint(0, 10) > 1:
                B += np.random.normal(0, 0.05)
                C += np.random.normal(0, 0.05)
        elif rand_val < param['SharpCubic']:
            if rand_val < param['KeysCubic']: # Keys Cubic
                B = np.random.uniform(0, 2 / 3) + np.random.normal(0, 1 / 3)
                C = 0.5 - B * 0.5
            elif rand_val < param['SoftCubic']: # Soft Cubic
                B = np.random.uniform(0.5, 1) + np.random.normal(0, 0.25)
                C = 1 - B
            else: # Sharp Cubic
                B = np.random.uniform(-0.75, -0.25) + np.random.normal(0, 0.25)
                C = B * -0.5
            # randomly alternate the kernel with 70%/90% probability
            if np.random.randint(0, 10) > (2 if rand_val < param['KeysCubic'] else 0):
                B += np.random.normal(0, 1 / 6)
                C += np.random.normal(0, 1 / 6)
        elif rand_val < param['ArtifactCubic']: # artifact Cubic
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

def random_filter(params, src, dw=None, dh=None, channel_first=False):
    param = params['random_filter']
    last = src
    sw = src.shape[-1 if channel_first else -2]
    sh = src.shape[-2 if channel_first else -3]
    if dw is None:
        dw = sw
    if dh is None:
        dh = sh
    scale = np.sqrt((dw * dh) / (sw * sh))
    # random number for scaling
    rand_val = np.random.randint(0, 100)
    if rand_val < param['NoScale']: # no scale
        rand_scale = 0
    elif rand_val < param['UpScale']: # up scale
        max_scale = max(0, np.log2(scale)) + param['max_scale']
        rand_scale = np.random.uniform(0, max_scale)
    elif rand_val < param['DownScale']: # down scale
        min_scale = min(0, np.log2(scale)) + param['min_scale']
        rand_scale = np.random.uniform(min_scale, 0)
    rand_scale = 2 ** rand_scale # [0.25, 1) + [1, 2)
    # random resize
    if rand_scale != 1: # random scale
        tw = int(sw * rand_scale + 0.5)
        th = int(sh * rand_scale + 0.5)
        # print('{}x{} => {}x{} => {}x{}'.format(sw, sh, tw, th, dw, dh))
        last = random_resize(params['random_resize'], last, tw, th,
            channel_first=channel_first)
    if rand_scale != 1 or dw != sw or dh != sh: # scale to target size
        last = random_resize(params['random_resize'], last, dw, dh,
            channel_first=channel_first)
    # return
    return last

def random_noise(param, src, matrix=None, channel_first=False):
    if param['noise_str'] <= 0.0:
        return src
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
    corrY = np.abs(np.random.normal(0.0, param['noise_corr']))
    corrY = 0 if corrY > param['noise_corr'] * 3 else corrY # no correlation if > sigma*3
    scaleY = np.abs(np.random.normal(0.0, param['noise_str'])) * (1 + corrY)
    corrC = np.abs(np.random.normal(0.0, param['noise_corr']))
    corrC = 0 if corrC > param['noise_corr'] * 3 else corrC # no correlation if > sigma*3
    scaleC = np.abs(np.random.normal(0.0, param['noise_str'])) * (1 + corrC)
    # noise type
    rand_val = np.random.randint(0, 100)
    # print('{}, Y: {}|{}, C: {}|{}'.format(rand_val, scaleY, corrY, scaleC, corrC))
    if rand_val < param['NoNoise']:
        pass
    if rand_val < param['RGB']: # RGB noise
        noise = noise_gen(shapeRGB, scaleY, corrY, channel_first=channel_first)
        last = last + noise
    elif rand_val < param['YUV444']: # YUV444 noise
        noiseY = noise_gen(shapeY, scaleY, corrY, channel_first=channel_first)
        noiseU = noise_gen(shapeY, scaleC, corrC, channel_first=channel_first)
        noiseV = noise_gen(shapeY, scaleC, corrC, channel_first=channel_first)
        noise = np.stack([noiseY, noiseU, noiseV], axis=0 if channel_first else -1)
        noise = zimg.convertFormat(noise, channel_first=channel_first, matrix_in=matrix, matrix='rgb')
        last = last + noise
    elif rand_val < param['Y']: # Y noise
        noiseY = noise_gen(shapeY, scaleY, corrY, channel_first=channel_first)
        noise = np.stack([noiseY] * 3, axis=0 if channel_first else -1)
        last = last + noise
    # return
    return last

def random_chroma(param, src, matrix=None, channel_first=False):
    last = src
    sw = src.shape[-1 if channel_first else -2]
    sh = src.shape[-2 if channel_first else -3]
    if matrix is None:
        matrix = ['BT709', 'ST170_M', 'BT2020_NCL'][np.random.randint(0, 3)]
    filters = (
        [{'filter': 'Bicubic', 'filter_a': 0, 'filter_b': 0.5}] * 3 +
        [{'filter': 'Bicubic', 'filter_a': 1/3, 'filter_b': 1/3}] * 2 +
        [{'filter': 'Bicubic', 'filter_a': 0.75, 'filter_b': 0.25},
        {'filter': 'Bicubic', 'filter_a': 1.0, 'filter_b': 0.0},
        {'filter': 'Point'}, {'filter': 'Bilinear'},
        {'filter': 'Lanczos', 'filter_a': 3}]
    )
    # 0: YUV420, MPEG-1 chroma placement
    # 1: YUV420, MPEG-2 chroma placement
    # 2~5: RGB
    rand_val = np.random.randint(0, 100)
    # chroma sub-sampling
    if rand_val < param['RGB']:
        pass
    elif rand_val < param['YUV420']:
        # convert RGB to YUV420
        last = zimg.convertFormat(last, channel_first=channel_first, matrix_in='rgb', matrix=matrix)
        lastY = last[0] if channel_first else last[:, :, 0]
        lastU = last[1] if channel_first else last[:, :, 1]
        lastV = last[2] if channel_first else last[:, :, 2]
        filter_params = filters[np.random.randint(0, len(filters))]
        resizer = zimg.Resizer.createScale(lastU, 0.5, **filter_params, channel_first=channel_first,
            roi_left=0 if rand_val % 2 == 0 else -0.5)
        lastU = resizer(lastU)
        lastV = resizer(lastV)
        # convert YUV420 to RGB
        filter_params = filters[np.random.randint(0, len(filters))]
        resizer = zimg.Resizer.create(lastU, sw, sh, **filter_params, channel_first=channel_first,
            roi_left=0 if rand_val % 2 == 0 else 0.25)
        lastU = resizer(lastU)
        lastV = resizer(lastV)
        last = np.stack((lastY, lastU, lastV), axis=0 if channel_first else -1)
        last = zimg.convertFormat(last, channel_first=channel_first, matrix_in=matrix, matrix='rgb')
    # return
    return last

def linear_resize(src, dw, dh, transfer, channel_first=False):
    last = src
    # convert to linear scale
    if transfer.upper() != 'LINEAR':
        last = zimg.convertFormat(last, channel_first=channel_first, transfer_in=transfer, transfer='LINEAR')
    # resize
    last = zimg.resize(last, dw, dh, 'Bicubic', 0, 0.5, channel_first=channel_first)
    # convert back to gamma-corrected scale
    if transfer.upper() != 'LINEAR':
        last = zimg.convertFormat(last, channel_first=channel_first, transfer_in='LINEAR', transfer=transfer)
    # return
    return last

def random_quantize(param, src, dtype=None, channel_first=False):
    if dtype is None:
        dtype = src.dtype
    last = src
    rand_val = np.random.randint(0, 100)
    # if needed, convert to 8-bit
    if rand_val >= param['NoQuant']:
        last = convert_dtype(last, np.uint8)
    # if needed, convert CHW to HWC
    if rand_val >= param['Quant8'] and channel_first:
        last = np.transpose(last, (1, 2, 0))
    # randomly encode image
    if rand_val < param['Quant8']:
        pass
    elif rand_val < param['WebP']: # WebP
        preset = list(webp.WebPPreset)
        preset = preset[np.random.randint(0, len(preset))]
        # random quality in [0, 100) with gamma correction
        # gamma > 1.0: bias towards small values
        # 0.0 < gamma < 1.0: bias towards big values
        gamma = param['webp_gamma']
        quality = np.random.uniform(0, 100 ** (1 / gamma)) ** gamma
        # print('WebP: preset={}, quality={}'.format(preset, quality))
        # encode and decode
        last = np.copy(last, order='C')
        pic = webp.WebPPicture.from_numpy(last)
        config = webp.WebPConfig.new(preset=preset, quality=quality, lossless=False)
        data = pic.encode(config)
        last = data.decode(color_mode=webp.WebPColorMode.RGB)
    elif rand_val < param['JPEG']: # JPEG
        subsampling = ['4:4:4'] * 3 + ['4:2:2', '4:2:0']
        subsampling = subsampling[np.random.randint(0, len(subsampling))]
        qtables = [None, None, 'web_low', 'web_high']
        qtables = qtables[np.random.randint(0, len(qtables))]
        if qtables is None:
            quality = 0
            while not (1 <= quality <= 100):
                quality = np.random.normal(param['jpeg_mean'], param['jpeg_std'])
            quality = int(quality + 0.5)
        else:
            quality = np.random.randint(1, 101)
        # print('JPEG: sub={}, qtables={}, quality={}'.format(subsampling, qtables, quality))
        # encode and decode
        with BytesIO() as buffer:
            im = Image.fromarray(last)
            im.save(buffer, 'JPEG', subsampling=subsampling, quality=quality, qtables=qtables)
            im = Image.open(buffer)
            last = np.array(im, copy=False)
    # if needed, convert HWC to CHW
    if rand_val >= param['Quant8'] and channel_first:
        last = np.transpose(last, (2, 0, 1))
    # convert to output dtype
    last = convert_dtype(last, dtype)
    # return
    return last

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
    if config.augment and np.random.randint(0, 2) > 0:
        img = np.transpose(img, (0, 2, 1))
    # random flipping with 25% probability each
    rand_val = np.random.randint(0, 4) if config.augment else 0
    if rand_val == 1:
        img = img[:, :, ::-1]
    elif rand_val == 2:
        img = img[:, ::-1, :]
    elif rand_val == 3:
        img = img[:, ::-1, ::-1]
    # convert to float32
    img2 = convert_dtype(img, np.float32)
    # random filter (input)
    transfer = [None] * 3 + ['BT470_M', 'IEC_61966_2_1', 'IEC_61966_2_1']
    transfer = transfer[np.random.randint(0, len(transfer))]
    matrix = ['BT709'] * 3 + ['ST170_M', 'BT2020_NCL']
    matrix = matrix[np.random.randint(0, len(matrix))]
    _input = img2
    # randomly convert to linear scale
    if transfer is not None:
        _input = zimg.convertFormat(_input, channel_first=channel_first, transfer_in=transfer, transfer='LINEAR')
    # random filtering with resizer
    _input = random_filter(config.params, _input,
        config.patch_width // config.scale, config.patch_height // config.scale,
        channel_first=channel_first)
    # random noise
    _input = random_noise(config.params['random_noise'], _input,
        matrix=matrix, channel_first=channel_first)
    # convert back to gamma-corrected scale
    if transfer is not None:
        _input = zimg.convertFormat(_input, channel_first=channel_first, transfer_in='LINEAR', transfer=transfer)
    # random chroma sub-sampling
    _input = random_chroma(config.params['random_chroma'], _input,
        matrix=matrix, channel_first=channel_first)
    # random quantize, gamma2linear, type conversion (input)
    if config.linear:
        _input = random_quantize(config.params['random_quantize'], _input,
            np.float32, channel_first=channel_first)
        _input = zimg.convertFormat(_input, channel_first=channel_first, transfer_in=config.transfer, transfer='LINEAR')
        _input = convert_dtype(_input, dtype)
    else:
        _input = random_quantize(config.params['random_quantize'], _input,
            dtype, channel_first=channel_first)
    # pre downscale and type conversion (label)
    _label = img2
    if config.linear:
        _label = zimg.convertFormat(_label, channel_first=channel_first, transfer_in=config.transfer, transfer='LINEAR')
    if pre_scale != 1:
        _label = linear_resize(_label, config.patch_width, config.patch_height,
            'LINEAR' if config.linear else config.transfer, channel_first=channel_first)
    _label = convert_dtype(_label, dtype)
    # return
    return _input, _label # CHW, dtype

def mixup(config, img1, img2, alpha=1.2, dtype=np.float32):
    # process and mixup in float32
    inter_dtype = dtype if dtype in [np.float16, np.float32, np.float64] else np.float32
    _input1, _label1 = pre_process(config, img1, inter_dtype)
    _input2, _label2 = pre_process(config, img2, inter_dtype)
    _lambda = np.random.beta(alpha, alpha)
    _input = _lambda * _input1 + (1 - _lambda) * _input2
    _label = _lambda * _label1 + (1 - _lambda) * _label2
    # convert to output dtype
    _input = convert_dtype(_input, dtype)
    _label = convert_dtype(_label, dtype)
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
            _input = input()
            if _input == 'Y':
                import shutil
                shutil.rmtree(config.save_dir)
                eprint('Removed: ' + config.save_dir)
            elif _input != 'n':
                import sys
                sys.exit()
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
    def process(config, ifiles, ofile):
        dtype = np.dtype(config.dtype)
        inputs = []
        labels = []
        for ifile in ifiles:
            try:
                im = Image.open(ifile)
                img = np.array(im, copy=False)
                _input, _label = pre_process(config, img, dtype)
                inputs.append(_input)
                labels.append(_label)
            except Exception as err:
                print('======\nError when processing {}\n{}\n------'.format(ifile, err))
                # fill zero for data with error
                _blank = np.zeros((3, config.patch_height, config.patch_width), dtype)
                inputs.append(_blank)
                labels.append(_blank)
        # CHW => NCHW
        inputs = np.stack(inputs, axis=0)
        labels = np.stack(labels, axis=0)
        np.savez_compressed(ofile, inputs=inputs, labels=labels)

    @staticmethod
    def process_mixup(config, ifiles, ifiles2, ofile):
        dtype = np.dtype(config.dtype)
        inputs = []
        labels = []
        for ifile, ifile2 in zip(ifiles, ifiles2):
            try:
                im = Image.open(ifile)
                img = np.array(im, copy=False)
                im2 = Image.open(ifile2)
                img2 = np.array(im2, copy=False)
                _input, _label = mixup(config, img, img2, dtype=dtype)
                inputs.append(_input)
                labels.append(_label)
            except Exception as err:
                print('======\nError when processing {}\n{}\n------'.format(ifile, err))
                # fill zero for data with error
                _blank = np.zeros((3, config.patch_height, config.patch_width), dtype)
                inputs.append(_blank)
                labels.append(_blank)
        # CHW => NCHW
        inputs = np.stack(inputs, axis=0)
        labels = np.stack(labels, axis=0)
        np.savez_compressed(ofile, inputs=inputs, labels=labels)

    @classmethod
    def run(cls, config, dataset):
        _dataset = dataset.copy()
        _dataset2 = dataset.copy()
        epochs = config.epochs
        epoch_steps = len(_dataset) // config.batch_size
        step_width = len(str(epoch_steps))
        # pre-shuffle the dataset
        if config.shuffle == 1:
            random.shuffle(_dataset)
            random.shuffle(_dataset2)
        # execute pre-process
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(config.processes) as executor:
            skipped = 0
            for epoch in range(epochs):
                # create directory for each epoch
                odir = os.path.join(config.save_dir, '{:0>{width}}'.format(epoch, width=len(str(epochs))))
                if not os.path.exists(odir):
                    print('Create directory: ', odir)
                    os.makedirs(odir)
                # randomly shuffle for each epoch
                if config.shuffle == 2:
                    random.shuffle(_dataset)
                    random.shuffle(_dataset2)
                # loop over the batches and append the calls
                futures = []
                for step in range(epoch_steps):
                    ofile = os.path.join(odir, '{:0>{width}}.npz'.format(step, width=step_width))
                    # skip existing files
                    if not os.path.exists(ofile):
                        if skipped > 0:
                            print('Skipped {} existed output files'.format(skipped))
                            skipped = 0
                        begin = step * config.batch_size
                        end = begin + config.batch_size
                        ifiles = _dataset[begin : end]
                        if config.mixup:
                            ifiles2 = _dataset2[begin : end]
                            futures.append(executor.submit(cls.process_mixup, config, ifiles, ifiles2, ofile))
                        else:
                            futures.append(executor.submit(cls.process, config, ifiles, ofile))
                    else:
                        skipped += 1
                # execute the calls
                step = 0
                tick = time()
                for future in futures:
                    future.result()
                    # log speed every log_freq, always log speed at the end of each epoch
                    if (config.log_freq > 0 and step % config.log_freq == 0) or (step == len(futures) - 1):
                        tock = time()
                        speed = (config.batch_size * config.log_freq) / max(1e-9, tock - tick)
                        print('Epoch {} Step {}: {} samples/sec'.format(epoch, step, speed))
                        tick = time()
                    step += 1

    def __call__(self):
        self.initialize(self.config)
        dataset = self.get_dataset(self.config)
        self.run(self.config, dataset)

def main(argv):
    import argparse
    argp = argparse.ArgumentParser(argv[0])
    argp.add_argument('input_dir')
    argp.add_argument('save_dir')
    argp.add_argument('--params', required=True)
    argp.add_argument('--random-seed', type=int)
    argp.add_argument('--batch-size', type=int, default=1)
    argp.add_argument('--epochs', type=int, default=1)
    argp.add_argument('--shuffle', type=int, default=2) # 0: no shuffle, 1: shuffle once, 2: shuffle every epoch
    argp.add_argument('--log-freq', type=int, default=1000)
    argp.add_argument('--processes', type=int, default=8)
    argp.add_argument('--dtype', default='float16')
    bool_argument(argp, 'test', False)
    bool_argument(argp, 'augment', True)
    bool_argument(argp, 'pre-down', False)
    bool_argument(argp, 'linear', False)
    bool_argument(argp, 'mixup', False)
    argp.add_argument('--scale', type=int, default=1)
    argp.add_argument('--patch-width', type=int, default=256)
    argp.add_argument('--patch-height', type=int, default=256)
    argp.add_argument('--transfer', default='IEC_61966_2_1')
    # parse
    args = argp.parse_args(argv[1:])
    # force argument
    if args.test:
        args.augment = False
        args.linear = False
        args.mixup = False
    # load json
    import json
    with open(args.params) as fp:
        args.params = json.load(fp)
    # run data writer
    writer = DataWriter(args)
    writer()

if __name__ == '__main__':
    import sys
    main(sys.argv)
