import os

import pandas as pd

os.environ['TPU_NAME'] = 'local'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['NEXT_PLUGGABLE_DEVICE_USE_C_API'] = 'true'
os.environ['TF_PLUGGABLE_DEVICE_LIBRARY_PATH'] = '/lib/libtpu.so'

import keras

import tensorflow as tf

from metrics import SSIM, PSNR
from models.masked_sinogram_autoencoder import MaskedSinogramAutoencoder
from models.hybrid_model import HybridModel
from keras.optimizers.schedules import CosineDecay

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

interpolation = "bilinear"


@tf.function
def transform_denoise(data, fbp):
    sinogram, gt = data
    sinogram = tf.expand_dims(sinogram - 0.030857524, -1) / 31.87962
    sinogram = tf.image.resize(sinogram, (1024, 513), method=interpolation)
    gt = tf.expand_dims(gt - 0.16737686, -1) / 0.11505456
    fbp = tf.expand_dims(fbp - 0.16737686, -1) / 0.11505456
    fbp = tf.image.resize(fbp, (512, 512), method=interpolation)

    return (sinogram, fbp), gt


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


feature_desc_fbp = {
    'fbp': tf.io.FixedLenFeature([], tf.string),
}


def _parse_example_fbp(example_proto):
    res = tf.io.parse_single_example(example_proto, feature_desc_fbp)
    fbp = tf.io.parse_tensor(res['fbp'], out_type=tf.float32)
    fbp.set_shape((362, 362))
    return fbp


train_ds_denoise = tf.data.Dataset.zip(
    (tf.data.TFRecordDataset('gs://computed-tomography-ai/data/lodopab/lodopab_train.tfrecord').map(_parse_example),
     tf.data.TFRecordDataset('gs://computed-tomography-ai/data/lodopab-fbp/lodopab_train_fbp.tfrecord')
     .map(_parse_example_fbp))
).batch(GLOBAL_BATCH_SIZE).map(transform_denoise)

val_ds_denoise = tf.data.Dataset.zip(
    (tf.data.TFRecordDataset('gs://computed-tomography-ai/data/lodopab-valtestchallenge/lodopab_validation.tfrecord')
     .map(_parse_example),
     tf.data.TFRecordDataset('gs://computed-tomography-ai/data/lodopab-fbp/lodopab_val_fbp.tfrecord')
     .map(_parse_example_fbp))
).batch(GLOBAL_BATCH_SIZE).map(transform_denoise)

test_ds_denoise = tf.data.Dataset.zip(
    (tf.data.TFRecordDataset('gs://computed-tomography-ai/data/lodopab-valtestchallenge/lodopab_test.tfrecord')
     .map(_parse_example),
     tf.data.TFRecordDataset('gs://computed-tomography-ai/data/lodopab-fbp/lodopab_test_fbp'
                             '.tfrecord').map(_parse_example_fbp))
).batch(GLOBAL_BATCH_SIZE).map(transform_denoise)

with strategy.scope():
    mae = MaskedSinogramAutoencoder(
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
        dose=-1
    )

    model = HybridModel(
        mae,
        num_mask=0,
        dec_dim=1024,
        dec_mlp_units=1024 * 4,
        dec_layers=4,
        output_patch_width=16,
        output_patch_height=16,
        output_x_patches=32,
        output_y_patches=32,
        final_shape=(362, 362, 1),
        dec_embedding_type='learned'
    )

    # build the MAE and print out the number of parameters
    mae(
        tf.random.normal(shape=(1, 1024, 513, 1))
    )
    mae.load_weights('mae_model.weights.h5')

    lr = CosineDecay(
        initial_learning_rate=1e-6,
        warmup_target=1e-5,
        alpha=1e-5,
        warmup_steps=35840 / GLOBAL_BATCH_SIZE,
        decay_steps=69 * 35840 / GLOBAL_BATCH_SIZE,
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

    # build the model and print out the number of parameters=
    model(
        (
            tf.random.normal(shape=(1, 1024, 513, 1)),
            tf.random.normal(shape=(1, 512, 512, 1))
        )
    )

    model.summary()

    print()

    # train the model
    history = model.fit(train_ds_denoise, validation_data=val_ds_denoise, epochs=70)
    model.save_weights('hybrid_model.weights.h5')
    model.save('hybrid_model.keras')
    training_df = pd.DataFrame(data=history.history)
    training_df.to_csv('hybrid_model_training.csv')

    # # unfreeze mae and finetune
    # model.autoencoder.trainable = True
    # model.compile(
    #     optimizer=keras.optimizers.AdamW(weight_decay=1e-6, learning_rate=1e-5, beta_1=0.9, beta_2=0.95),
    #     loss="mse",
    #     metrics=[
    #         "mean_squared_error",
    #         "mean_absolute_error",
    #         PSNR(rescaling=True, mean=0.16737686, std=0.11505456),
    #         SSIM(rescaling=True, mean=0.16737686, std=0.11505456)
    #     ]
    # )
    # model.fit(train_ds_denoise, validation_data=val_ds_denoise, epochs=1, steps_per_epoch=1, validation_steps=1)
