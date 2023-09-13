import tensorflow as tf

import keras_core as kc
import keras_core.ops as ops


def preprocess_data(sinogram, gt, resize_img=True, expand_dims=True):
    # some rescaling
    if expand_dims:
        sinogram = ops.expand_dims(sinogram, axis=-1)

    sinogram = (sinogram - 0.030857524) / 0.023017514
    sinogram = ops.image.resize(sinogram, (1024, 513), method="bilinear")

    if expand_dims:
        gt = ops.expand_dims(gt, axis=-1)

    gt = (gt - 0.16737686) / 0.11505456
    if resize_img:
        gt = ops.image.resize(gt, (512, 512))

    return sinogram, gt


def add_noise(img, dose=4096):
    img = dose * ops.exp(-img * 81.35858)

    img = img + kc.random.normal(shape=ops.shape(img), mean=0.0, stddev=img ** 0.5, dtype="float32")
    img = ops.clip(img / dose, 0.1 / dose, tf.float32.max)
    img = -ops.log(img) / 81.35858
    return img


def process_sinogram(sinogram):
    sinogram = ops.clip(sinogram, 0, 10)
    sinogram = sinogram[:, ::-1, ::-1, :] * 0.46451485

    sinogram = (sinogram - 0.030857524) / 0.023017514
    sinogram = ops.image.resize(sinogram, (1024, 513))
    return sinogram
