import keras.ops as ops
from keras.layers import *
from keras.models import *
import keras

from layers import Patches, PatchEncoder, PatchDecoder, TransformerDecoder


class HybridModel(Model):
    def __init__(
            self,
            autoencoder,
            num_mask=0,
            dec_dim=256,
            dec_layers=8,
            dec_heads=16,
            dec_mlp_units=512,
            output_patch_height=16,
            output_patch_width=16,
            output_x_patches=16,
            output_y_patches=16,
            final_shape=(362, 362, 1),
            divide_heads=True,
            dec_embedding_type='learned',
            name=None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.num_mask = num_mask

        self.dec_dim = dec_dim
        self.dec_layers = dec_layers
        self.dec_heads = dec_heads
        self.dec_mlp_units = dec_mlp_units

        self.output_patch_height = output_patch_height
        self.output_patch_width = output_patch_width
        self.output_x_patches = output_x_patches
        self.output_y_patches = output_y_patches

        self.divide_heads = divide_heads

        self.autoencoder = autoencoder
        self.inp_shape = autoencoder.inp_shape

        self.num_patches = autoencoder.num_patches

        # building patch encoder layers
        self.patches = Patches(
            self.output_patch_width,
            self.output_patch_height,
            name='dec_patches'
        )
        self.patch_encoder = PatchEncoder(
            self.output_y_patches * self.output_x_patches,
            self.dec_dim,
            embedding_type=dec_embedding_type,
            name='dec_projection'
        )

        # building decoder layers
        self.decoder = [
            TransformerDecoder(
                project_dim=self.dec_dim,
                num_heads=self.dec_heads,
                mlp_dim=self.dec_mlp_units,
                mlp_dropout=self.autoencoder.dropout,
                attention_dropout=self.autoencoder.dropout,
                activation=self.autoencoder.activation,
                divide_heads=divide_heads,
                layer_norm_epsilon=self.autoencoder.layer_norm_epsilon,
                name=f'dec_block_{i}'
            ) for i in range(self.dec_layers)
        ]

        self.norm_layer = LayerNormalization(epsilon=self.autoencoder.layer_norm_epsilon, name=f'output_norm')

        self.dense = Dense(self.output_patch_height * self.output_patch_width, name='output_projection')
        self.patch_decoder = PatchDecoder(
            self.output_patch_width,
            self.output_patch_height,
            self.output_x_patches,
            self.output_y_patches
        )

        # resize so the model will use the legit resolution and give back the legit PSNR
        self.resize = Resizing(
            final_shape[0], final_shape[1]
        )

        # configure autoencoder
        self.autoencoder.trainable = False
        self.autoencoder.patch_encoder.num_mask = self.num_mask

    def call(self, inputs, training=None, mask=None):
        x, y = inputs

        # Convert to patches and encode
        x = self.autoencoder.patches(x)
        x, _, _ = self.autoencoder.encoder_impl(x)

        y = self.patches(y)
        y = self.patch_encoder(y)

        for dec_block in self.decoder:
            y = dec_block((y, x))

        y = self.norm_layer(y)
        y = self.dense(y)
        y = self.patch_decoder(y)

        return self.resize(y)


if __name__ == "__main__":
    import tensorflow as tf

    from models.masked_sinogram_autoencoder import MaskedSinogramAutoencoder

    autoencoder = MaskedSinogramAutoencoder(
        enc_layers=4,
        dec_layers=1,
        sinogram_width=513,
        sinogram_height=1,
        input_shape=(1024, 513, 1),
        enc_dim=512,
        enc_mlp_units=2048,
        dec_dim=512,
        dec_mlp_units=2048,
        mask_ratio=0.75
    )

    model = HybridModel(
        autoencoder,
        num_mask=0,
        dec_dim=512,
        dec_layers=8,
        dec_heads=16,
        dec_mlp_units=512,
        output_patch_height=16,
        output_patch_width=16,
        output_x_patches=32,
        output_y_patches=32,
        final_shape=(362, 362, 1),
    )

    def transform(_):
        x = tf.random.normal((1, 1024, 513, 1))
        y = tf.random.normal((1, 10, 10, 1))

        return (x, tf.image.resize(y, (512, 512))), tf.image.resize(y, (362, 362))

    train_ds = tf.data.Dataset.random(1).map(transform)
    test_ds = tf.data.Dataset.random(1).map(transform)

    rand_indices = tf.argsort(
        tf.random.uniform(shape=(1, 1024)), axis=-1
    )
    model.call(
        (
            tf.random.normal(shape=(1, 1024, 513, 1)),
            tf.random.normal(shape=(1, 512, 512, 1))
        )
    )
    print(model.summary())

    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5), loss='mse')
    model.fit(train_ds, validation_data=test_ds, steps_per_epoch=100, validation_steps=10, epochs=1)
    