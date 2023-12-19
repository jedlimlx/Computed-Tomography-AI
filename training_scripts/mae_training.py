import os

os.environ['TPU_NAME'] = 'local'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['NEXT_PLUGGABLE_DEVICE_USE_C_API'] = 'true'
os.environ['TF_PLUGGABLE_DEVICE_LIBRARY_PATH'] = '/lib/libtpu.so'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf
import keras

from models.masked_sinogram_autoencoder import MaskedSinogramAutoencoder
import pandas as pd

cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
strategy = tf.distribute.TPUStrategy(cluster_resolver)
print("Number of accelerators: ", strategy.num_replicas_in_sync)

PER_REPLICA_BATCH_SIZE = 8
AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
GLOBAL_BATCH_SIZE = PER_REPLICA_BATCH_SIZE * REPLICAS
print(f'REPLICAS: {REPLICAS}')


@tf.function
def transform(sinogram, gt):
    sinogram = tf.expand_dims(sinogram - 0.030857524, -1) / 0.023017514
    sinogram = tf.image.resize(sinogram, (1024, 513))
    return sinogram


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
            .map(_parse_example)
            .batch(GLOBAL_BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
            .shuffle(100)
            .map(transform))

val_ds = (tf.data.TFRecordDataset('gs://kds-bb472b8b8411cc589272ec67c8152102bcf14429f2c5e7af07ab24aa'
                                  '/lodopab_validation.tfrecord')
          .map(_parse_example)
          .batch(GLOBAL_BATCH_SIZE)
          .prefetch(tf.data.AUTOTUNE)
          .map(transform))

with strategy.scope():
    model = MaskedSinogramAutoencoder(
        enc_layers=4,
        dec_layers=1,
        sinogram_width=513,
        sinogram_height=1,
        input_shape=(1024, 513, 1),
        enc_dim=1024,
        enc_mlp_units=1024 * 4,
        dec_dim=1024,
        dec_mlp_units=1024 * 4,
        mask_ratio=0.75,
        radon_transform=None,
        dose=-1
    )

    model.compile(optimizer=keras.optimizers.AdamW(weight_decay=1e-5, learning_rate=1e-4), loss='mse')

    # build the model
    model(keras.random.normal(shape=(1, 1024, 513, 1)))
    model.summary()

    # training the model with the perlin noise
    # print("\nTraining with perlin noise...")
    # model.fit(ds, validation_data=val_ds, epochs=20, steps_per_epoch=560, validation_steps=56)

    # model.save_weights("mae_model.weights.h5")
    # model.save("mae_model.keras")

    model.compile(optimizer=keras.optimizers.AdamW(weight_decay=1e-5, learning_rate=1e-4), loss='mse')

    # model.load_weights("../input/ctransformer-masked-sinogram-autoencoder/mae_model.weights.h5")

    print("\nTraining with real data...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=100)

    model.save_weights("mae_model.weights.h5")
    model.save("mae_model.keras")

    df = pd.DataFrame(data=history.history)
    df.to_csv("mae_model.csv")
