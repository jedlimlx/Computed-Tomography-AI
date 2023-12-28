import os

os.environ['TPU_NAME'] = 'local'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['NEXT_PLUGGABLE_DEVICE_USE_C_API'] = 'true'
os.environ['TF_PLUGGABLE_DEVICE_LIBRARY_PATH'] = '/lib/libtpu.so'

import keras
from keras.optimizers.schedules import CosineDecay
import tensorflow as tf
from metrics import PSNR, SSIM
from models import DirectReconstructionModel
import pandas as pd

PER_REPLICA_BATCH_SIZE = 8
cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
strategy = tf.distribute.TPUStrategy(cluster_resolver)
print("Number of accelerators: ", strategy.num_replicas_in_sync)

AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
GLOBAL_BATCH_SIZE = PER_REPLICA_BATCH_SIZE * REPLICAS
print(f'REPLICAS: {REPLICAS}')

interpolation = "bilinear"


@tf.function
def transform_mae(sinogram, gt):
    sinogram = tf.expand_dims(sinogram - 0.030857524, -1) / 0.023017514
    sinogram = tf.image.resize(sinogram, (1024, 513))
    return sinogram


@tf.function
def transform_denoise(sinogram, gt):
    sinogram = tf.expand_dims(sinogram - 0.030857524, -1) / 0.023017514
    sinogram = tf.image.resize(sinogram, (1024, 513))

    gt = (gt - 0.16737686) / 0.11505456
    return sinogram, gt[..., tf.newaxis]


feature_desc = {
    'observation': tf.io.FixedLenFeature([], tf.string),
    'ground_truth': tf.io.FixedLenFeature([], tf.string)
}


def _parse_example(example_proto):
    res = tf.io.parse_single_example(example_proto, feature_desc)
    observation = tf.io.parse_tensor(res['observation'], out_type=tf.float32)
    ground_truth = tf.io.parse_tensor(res['ground_truth'], out_type=tf.float32)
    observation.set_shape((1000, 513))
    ground_truth.set_shape((362, 362))
    return observation, ground_truth


train_ds = (tf.data.TFRecordDataset('gs://computed-tomography-ai/data/lodopab/lodopab_train'
                                    '.tfrecord')
            .map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(GLOBAL_BATCH_SIZE)
            .map(transform_denoise, num_parallel_calls=tf.data.AUTOTUNE))
val_ds = (tf.data.TFRecordDataset('gs://computed-tomography-ai/data/lodopab-valtestchallenge/lodopab_validation'
                                  '.tfrecord')
          .map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(GLOBAL_BATCH_SIZE)
          .map(transform_denoise, num_parallel_calls=tf.data.AUTOTUNE))
test_ds = (tf.data.TFRecordDataset('gs://computed-tomography-ai/data/lodopab-valtestchallenge/lodopab_test'
                                   '.tfrecord')
           .map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
           .batch(GLOBAL_BATCH_SIZE)
           .map(transform_denoise, num_parallel_calls=tf.data.AUTOTUNE))


with strategy.scope():
    model = DirectReconstructionModel(
        input_shape=(1024, 513, 1),
        final_shape=(362, 362, 1),
        enc_layers=4,
        dec_layers=4,
        sinogram_width=513,
        sinogram_height=1,
        enc_dim=1024,
        enc_mlp_units=1024 * 4,
        dec_dim=1024,
        dec_mlp_units=1024 * 4,
        output_patch_width=16,
        output_patch_height=16,
        output_x_patches=32,
        output_y_patches=32,
        output_projection=True,
        divide_heads=True,
        layer_norm_epsilon=1e-5,
    )
    # print(strategy.run(tf.function(lambda x: model(x)), args=(keras.random.normal(shape=(64, 1024, 513, 1)),)))

    # lr = keras.optimizers.schedules.CosineDecay(1.6e-4, 4000)

    model(tf.random.normal((1, 1024, 513, 1)))

    lr = CosineDecay(
        initial_learning_rate=1e-6,
        warmup_target=1e-5,
        alpha=1e-6,
        warmup_steps=35840 / GLOBAL_BATCH_SIZE,
        decay_steps=99 * 35840 / GLOBAL_BATCH_SIZE,
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr),
        loss="mse",
        metrics=[
            "mean_squared_error",
            "mean_absolute_error",
            PSNR(rescaling=True, mean=0.16737686, std=0.11505456),
            SSIM(rescaling=True, mean=0.16737686, std=0.11505456)
        ]
    )
    model.summary()

    history = model.fit(train_ds, validation_data=val_ds, epochs=100)
    model.save_weights("direct_reconstruction.weights.h5")
    model.save("direct_reconstruction.keras")

    df = pd.DataFrame(data=history.history)
    df.to_csv('direct_reconstruction.csv')
