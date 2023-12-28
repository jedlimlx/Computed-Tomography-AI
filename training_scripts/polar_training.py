import os

import pandas as pd

from metrics import PSNR, SSIM

os.environ['TPU_NAME'] = 'local'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['NEXT_PLUGGABLE_DEVICE_USE_C_API'] = 'true'
os.environ['TF_PLUGGABLE_DEVICE_LIBRARY_PATH'] = '/lib/libtpu.so'

import keras
import tensorflow as tf

from models.masked_sinogram_autoencoder import MaskedSinogramAutoencoder
from models.polar_transformer import PolarTransformer
from keras.optimizers.schedules import CosineDecay

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

@tf.function
def transform_denoise(data, fbp):
    sinogram, gt = data
    sinogram = tf.expand_dims(sinogram - 0.030857524, -1) / 0.023017514
    sinogram = tf.image.resize(sinogram, (1024, 513), method='bilinear')
    gt = tf.expand_dims(gt - 0.16737686, -1) / 0.11505456
    fbp = tf.expand_dims(fbp - 0.16737686, -1) / 0.11505456

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

    model = PolarTransformer(
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
    model((tf.random.normal((1, 1024, 513, 1)), tf.random.normal((1, 362, 362, 1))))

    lr = CosineDecay(
        initial_learning_rate=1e-6,
        warmup_target=1e-5,
        alpha=1e-6,
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

    training_history = model.fit(train_ds_denoise, validation_data=val_ds_denoise, epochs=1, steps_per_epoch=20, validation_steps=20)
    training_df = pd.DataFrame(data=training_history.history)
    training_df.to_csv("polar_transformer_training.csv")

    # unfreeze mae and finetune
    # autoencoder.trainable = True
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
    # finetuning_history = model.fit(train_ds_denoise, validation_data=val_ds_denoise, epochs=1, steps_per_epoch=1, validation_steps=1)
    # finetuning_df = pd.DataFrame(data=finetuning_history.history)
    # finetuning_df.to_csv("polar_transformer_finetuning.csv")

    model.save_weights("polar_transformer.weights.h5")
    model.save("polar_transformer.keras")
