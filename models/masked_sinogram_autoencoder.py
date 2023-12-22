import tensorflow as tf

import keras
import keras.ops as ops

from keras.layers import *
from keras.models import *

from layers import Patches, PatchDecoder, TransformerEncoder, SinePositionEncoding, PositionEmbedding

from functools import partial

from utils import add_noise
from utils.data import process_sinogram


class SinogramPatchEncoder(Layer):
    def __init__(
            self,
            num_patches,
            projection_dim,
            mask_proportion=0.75,
            embedding_type='sin_cos',
            name=None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.mask_proportion = mask_proportion

        self.embedding_type = embedding_type

        self.num_mask = int(num_patches * mask_proportion)

        # projection
        self.projection = Dense(units=projection_dim)

        # positional encoding
        if embedding_type == 'learned':
            self.position_embedding = PositionEmbedding(num_patches)
        else:
            self.position_embedding = SinePositionEncoding(num_patches)

    def build(self, input_shape):
        _, depth, area = input_shape
        self.mask_token = self.add_weight(shape=(1, area), initializer='random_uniform', name=f'{self.name}_mask_token')

    def call(self, patch, mask_indices=None, unmask_indices=None, **kwargs):
        batch_size = ops.shape(patch)[0]

        # encoding + positional embedding
        encoded = self.projection(patch)
        embedding = self.position_embedding(encoded)

        encoded = encoded + embedding

        if mask_indices is None or unmask_indices is None:
            mask_indices, unmask_indices = self.get_random_indices(batch_size)

        # The encoder input is the unmasked patch embeddings. Here we gather
        # all the patches that should be unmasked.
        unmasked_embeddings = tf.gather(
            encoded, unmask_indices, axis=1, batch_dims=1  # todo use ops.take when it supports batch_dims
        )  # (B, unmask_numbers, projection_dim)

        # Get the unmasked and masked position embeddings. We will need them
        # for the decoder.
        unmasked_positions = tf.gather(
            embedding, unmask_indices, axis=1, batch_dims=1
        )  # (B, unmask_numbers, projection_dim)
        masked_positions = tf.gather(
            embedding, mask_indices, axis=1, batch_dims=1
        )  # (B, mask_numbers, projection_dim)

        # Repeat the mask token number of mask times.
        # Mask tokens replace the masks of the image.
        if self.num_mask != 0:
            mask_tokens = ops.repeat(self.mask_token, repeats=self.num_mask, axis=0)
            mask_tokens = ops.repeat(
                ops.expand_dims(
                    mask_tokens, axis=0
                ), repeats=batch_size, axis=0
            )

            # Get the masked embeddings for the tokens.
            masked_embeddings = self.projection(mask_tokens) + masked_positions
        else:
            masked_embeddings = ops.zeros((batch_size, 0, self.projection_dim), dtype=self.dtype)

        return (
            unmasked_embeddings,  # Input to the encoder.
            masked_embeddings,  # First part of input to the decoder.
            unmasked_positions,  # Added to the encoder outputs.
            mask_indices,  # The indices that were masked.
            unmask_indices,  # The indices that were unmasked.
        )

    def get_random_indices(self, batch_size):
        # Create random indices from a uniform distribution and then split
        # it into mask and unmask indices.
        rand_indices = ops.argsort(
            keras.random.uniform(shape=(batch_size, self.num_patches)), axis=-1
        )
        mask_indices = rand_indices[:, :self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask:]
        return mask_indices, unmask_indices

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,
            'embedding_type': self.embedding_type,
            'mask_proportion': self.mask_proportion
        })
        return cfg


class MaskedSinogramAutoencoder(Model):
    def __init__(
            self,
            input_shape=(256, 256, 1),
            sinogram_height=1,
            sinogram_width=256,
            enc_dim=256,
            enc_layers=8,
            enc_mlp_units=512,
            enc_heads=16,
            dec_dim=256,
            dec_layers=8,
            dec_heads=16,
            dec_mlp_units=512,
            dropout=0.,
            activation='gelu',
            mask_ratio=0.75,
            divide_heads=True,
            radon_transform=None,
            dose=-1,
            denoise=False,
            name='mae',
            **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.radon_transform = radon_transform

        self.num_patches = int(input_shape[1] / sinogram_width * input_shape[0] / sinogram_height)

        self.dose = dose
        self.denoise = denoise

        self.inp_shape = input_shape
        self.sinogram_height = sinogram_height
        self.sinogram_width = sinogram_width

        self.enc_dim = enc_dim
        self.enc_layers = enc_layers
        self.enc_mlp_units = enc_mlp_units
        self.enc_heads = enc_heads

        self.dec_dim = dec_dim
        self.dec_layers = dec_layers
        self.dec_heads = dec_heads
        self.dec_mlp_units = dec_mlp_units

        self.dropout = dropout
        self.activation = activation
        self.mask_ratio = mask_ratio

        self.divide_heads = divide_heads

        if input_shape[0] % sinogram_height != 0 or input_shape[1] % sinogram_width != 0:
            raise ValueError("Cannot divide image into even patches")

        # patch encoder
        self.patches = Patches(sinogram_width, sinogram_height, f'{name}_patches')
        self.patch_encoder = SinogramPatchEncoder(
            self.num_patches,
            enc_dim,
            mask_proportion=mask_ratio,
            name=f'{name}_enc_projection'
        )

        # building encoder layer
        self.enc_norm = LayerNormalization(epsilon=1e-6, name=f'{name}_enc_norm')

        self.encoder = [
            TransformerEncoder(
                project_dim=enc_dim,
                num_heads=enc_heads,
                mlp_dim=enc_mlp_units,
                mlp_dropout=dropout,
                attention_dropout=dropout,
                activation=activation,
                divide_heads=divide_heads,
                name=f'{name}_enc_block_{i}'
            ) for i in range(self.enc_layers)
        ]

        # building decoder layers
        self.dec_norm = LayerNormalization(epsilon=1e-6, name=f'{name}_dec_norm')

        self.decoder_projection = Dense(dec_dim, name=f'{name}_dec_projection')

        self.decoder = [
            TransformerEncoder(
                project_dim=dec_dim,
                num_heads=dec_heads,
                mlp_dim=dec_mlp_units,
                mlp_dropout=dropout,
                attention_dropout=dropout,
                activation=activation,
                name=f'{name}_dec_block_{i}'
            ) for i in range(dec_layers)
        ]

        self.output_projection = Dense(sinogram_width * sinogram_height, name=f'{name}_output_projection')

        self.depatchify = PatchDecoder(
            sinogram_width, sinogram_height,
            int(input_shape[1] / sinogram_width),
            int(input_shape[0] / sinogram_height),
            name=f'{name}_depatchify'
        )

    def encoder_impl(self, patches):
        # Encode the patches.
        (
            unmasked_embeddings,
            masked_embeddings,
            unmasked_positions,
            mask_indices,
            unmask_indices,
        ) = self.patch_encoder(patches)

        # Pass the unmasked patches to the encoder.
        encoder_outputs = unmasked_embeddings
        for block in self.encoder:
            encoder_outputs = block(encoder_outputs)

        encoder_outputs = self.enc_norm(encoder_outputs)

        # Create the decoder inputs.
        encoder_outputs = encoder_outputs + unmasked_positions
        decoder_inputs = ops.concatenate([encoder_outputs, masked_embeddings], axis=1)

        return decoder_inputs, mask_indices, unmask_indices

    def call(self, inputs, denoised_inputs=None, training=None, mask=None):
        patches = self.patches(inputs)

        if denoised_inputs is None:
            denoised_patches = patches
        else:
            denoised_patches = self.patches(denoised_inputs)

        decoder_inputs, mask_indices, unmask_indices = self.encoder_impl(patches)

        # Decode the inputs.
        decoder_outputs = self.decoder_projection(decoder_inputs)
        for block in self.decoder:
            decoder_outputs = block(decoder_outputs)
        decoder_outputs = self.dec_norm(decoder_outputs)

        decoder_outputs = self.output_projection(decoder_outputs)

        decoder_patches = self.depatchify(decoder_outputs)

        return denoised_patches, decoder_patches, mask_indices, unmask_indices

    def train_step(self, data):
        if self.radon_transform is not None:
            sinogram = self.radon_transform(data, training=False)

            if self.dose > 0:
                noised_sinogram = add_noise(sinogram, dose=self.dose)
            else:
                noised_sinogram = sinogram

            sinogram = process_sinogram(sinogram)
            noised_sinogram = process_sinogram(noised_sinogram)

            data = sinogram
            noised_data = noised_sinogram

            if self.denoise:
                data = noised_sinogram
        else:
            noised_data = data

        with tf.GradientTape() as tape:
            patches, decoder_patches, mask_indices, unmasked_indices = self(noised_data, denoised_inputs=data)
            loss_patch = tf.gather(patches, mask_indices, axis=1, batch_dims=1)
            loss_output = decoder_patches[:, int(self.num_patches * (1 - self.mask_ratio)):]
            total_loss = self.compute_loss(y=loss_patch, y_pred=loss_output)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(total_loss)
            else:
                metric.update_state(y=loss_patch, y_pred=loss_output)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        patches, decoder_patches, mask_indices, unmasked_indices = self(data)
        loss_patch = tf.gather(patches, mask_indices, axis=1, batch_dims=1)
        loss_output = decoder_patches[:, int(self.num_patches * (1 - self.mask_ratio)):]

        loss = self.compute_loss(y=loss_patch, y_pred=loss_output)

        # Update the metrics.
        # Metrics are configured in `compile()`.
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y=loss_patch, y_pred=loss_output)

        return {m.name: m.result() for m in self.metrics}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        norm_cls = deserialize(config['norm']).__class__
        del config['norm']['config']['name']
        norm = partial(norm_cls, **config['norm']['config'])
        del config['norm']

        return cls(**config, norm=norm)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'input_shape': self.inp_shape,
            'sinogram_height': self.sinogram_height,
            'sinogram_width': self.sinogram_width,
            'enc_dim': self.enc_dim,
            'enc_layers': self.enc_layers,
            'enc_mlp_units': self.enc_mlp_units,
            'enc_heads': self.enc_heads,
            'dec_dim': self.dec_dim,
            'dec_layers': self.dec_layers,
            'dec_heads': self.dec_heads,
            'dec_mlp_units': self.dec_mlp_units,
            'dropout': self.dropout,
            'activation': self.activation,
            'mask_ratio': self.mask_ratio
        })

        return cfg


if __name__ == "__main__":
    import os

    os.environ["KERAS_BACKEND"] = "tensorflow"

    import keras

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
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=1e-5), loss='mse')
    print(model.call(keras.random.normal(shape=(1, 1024, 513, 1))))
    model.summary()

    model.save_weights("model.weights.h5")
    model.save("model.keras")
