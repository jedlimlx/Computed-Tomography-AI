import keras_core.ops as ops
from keras_core.layers import *
from keras_core.models import *

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

        self.autoencoder = autoencoder
        self.inp_shape = autoencoder.inp_shape

        self.num_patches = autoencoder.num_patches

        # building encoder layers
        self.patches = Patches(
            self.output_patch_width,
            self.output_patch_height,
            name='dec_patches'
        )
        self.patch_encoder = PatchEncoder(
            self.output_y_patches * self.output_x_patches,
            self.dec_dim,
            embedding_type='learned',
            name='dec_projection'
        )

        self.dec_blocks = [
            TransformerDecoder(
                self.dec_heads,
                self.autoencoder.enc_dim,
                self.dec_dim,
                mlp_units=self.dec_mlp_units,
                num_patches=self.num_patches,
                dropout=self.autoencoder.dropout,
                activation=self.autoencoder.activation,
                name=f'dec_block_{i}'
            ) for i in range(self.dec_layers)
        ]

        self.norm_layer = self.norm(name='output_norm')

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

        # freeze autoencoder weights
        self.autoencoder.trainable = False

    def call(self, inputs, training=None, mask=None):
        x, mask_indices, unmask_indices, y = inputs

        x = self.autoencoder.patches(x)

        self.autoencoder.patch_encoder.num_mask = self.num_mask

        (
            unmasked_embeddings,
            masked_embeddings,
            unmasked_positions,
            mask_indices,
            unmask_indices,
        ) = self.autoencoder.patch_encoder(x, mask_indices, unmask_indices)

        # Pass the unmaksed patches to the encoder.
        encoder_outputs = unmasked_embeddings

        for enc_block in self.autoencoder.enc_blocks:
            encoder_outputs = enc_block(encoder_outputs)

        encoder_outputs = self.autoencoder.enc_norm(encoder_outputs)

        # Create the decoder inputs.
        encoder_outputs = encoder_outputs + unmasked_positions
        x = ops.concatenate([encoder_outputs, masked_embeddings], axis=1)

        y = self.patches(y)
        y = self.patch_encoder(y)

        for dec_block in self.dec_blocks:
            y = dec_block((y, x))

        y = self.norm_layer(y)
        y = self.dense(y)
        y = self.patch_decoder(y)

        return self.resize(y)


if __name__ == "__main__":
    import keras_core

    model = MaskedSinogramAutoencoder(
        enc_layers=4,
        dec_layers=1,
        sinogram_width=513,
        sinogram_height=1,
        input_shape=(1024, 513, 1),
        enc_dim=256,
        enc_mlp_units=2048,
        dec_dim=256,
        dec_mlp_units=2048,
        mask_ratio=0.75
    )
    model.compile(optimizer=keras_core.optimizers.AdamW(learning_rate=5e-5, weight_decay=1e-5), loss='mse')
    model.call(keras_core.random.normal(shape=(1, 1024, 513, 1)))
    model.summary()