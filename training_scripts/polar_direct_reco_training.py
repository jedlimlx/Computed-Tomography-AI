import os

from models.polar_direct_reco import PolarDirectReco

os.environ['TPU_NAME'] = 'local'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['NEXT_PLUGGABLE_DEVICE_USE_C_API'] = 'true'
os.environ['TF_PLUGGABLE_DEVICE_LIBRARY_PATH'] = '/lib/libtpu.so'

import keras
import tensorflow as tf
from metrics import PSNR, SSIM
from models import DirectReconstructionModel, MaskedSinogramAutoencoder
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


train_ds = (tf.data.TFRecordDataset('gs://kds-febc291acaf8a01d21fe4181d8835d0cb95a786faae57be48addb7c5/lodopab_train'
                                    '.tfrecord')
            .map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(GLOBAL_BATCH_SIZE)
            .map(transform_denoise, num_parallel_calls=tf.data.AUTOTUNE))
val_ds = (tf.data.TFRecordDataset('gs://kds-bb472b8b8411cc589272ec67c8152102bcf14429f2c5e7af07ab24aa/lodopab_validation'
                                  '.tfrecord')
          .map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(GLOBAL_BATCH_SIZE)
          .map(transform_denoise, num_parallel_calls=tf.data.AUTOTUNE))
test_ds = (tf.data.TFRecordDataset('gs://kds-bb472b8b8411cc589272ec67c8152102bcf14429f2c5e7af07ab24aa/lodopab_test'
                                   '.tfrecord')
           .map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
           .batch(GLOBAL_BATCH_SIZE)
           .map(transform_denoise, num_parallel_calls=tf.data.AUTOTUNE))


with strategy.scope():
    autoencoder = MaskedSinogramAutoencoder(
        enc_layers=4,
        dec_layers=1,
        sinogram_width=513,
        sinogram_height=1,
        input_shape=(1024, 513, 1),
        enc_dim=1024,
        enc_mlp_units=4096,
        dec_dim=1024,
        dec_mlp_units=4096,
        mask_ratio=0.75,
        dose=-1,
    )

    model = PolarDirectReco(
        autoencoder,
        out_shape=(362, 362),
        dec_blocks=4,
        dec_dim=1024,
        dec_heads=16,
        dropout=0.,
        layer_norm_epsilon=1e-5,
        ipt_when_training=False,
        ipt_when_testing=True,
    )
    autoencoder(tf.random.normal((1, 1024, 513, 1)))
    autoencoder.load_weights('mae_model.weights.h5')
    model(tf.random.normal((1, 1024, 513, 1)))
    # print(strategy.run(tf.function(lambda x: model(x)), args=(keras.random.normal(shape=(64, 1024, 513, 1)),)))

    # lr = keras.optimizers.schedules.CosineDecay(1.6e-4, 4000)

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
        loss='mse',
        metrics=[PSNR(mean=0.16737686, std=0.11505456), SSIM(mean=0.16737686, std=0.11505456)]
    )
    model.summary()

    history = model.fit(train_ds, validation_data=val_ds, epochs=70)
    model.save_weights("polar_direct_reco.weights.h5")
    model.save("polar_direct_reco.keras")

    training_df = pd.DataFrame(data=history.history)
    training_df.to_csv('polar_direct_reco_training.csv')
