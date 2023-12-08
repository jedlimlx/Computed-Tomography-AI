from keras.layers import *
from keras.models import *

from layers import PatchEncoder, PatchDecoder, Patches, TransformerEncoder


class DirectReconstructionModel(Model):
    def __init__(
            self,
            input_shape=(256, 256, 1),
            final_shape=(362, 362, 1),
            sinogram_height=1,
            sinogram_width=256,
            enc_dim=256,
            enc_layers=8,
            enc_mlp_units=512,
            enc_heads=16,
            dec_projection=True,
            dec_dim=256,
            dec_layers=8,
            dec_heads=16,
            dec_mlp_units=512,
            output_projection=False,
            output_patch_height=16,
            output_patch_width=16,
            output_x_patches=16,
            output_y_patches=16,
            dropout=0.,
            layer_norm_epsilon=1e-5,
            activation='gelu',
            divide_heads=True,
            name='ctransformer0',
            **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.inp_shape = input_shape
        self.sinogram_height = sinogram_height
        self.sinogram_width = sinogram_width

        self.output_patch_height = output_patch_height
        self.output_patch_width = output_patch_width
        self.output_x_patches = output_x_patches
        self.output_y_patches = output_y_patches

        self.enc_dim = enc_dim
        self.enc_layers = enc_layers
        self.enc_mlp_units = enc_mlp_units
        self.enc_heads = enc_heads

        self.dec_projection = dec_projection
        self.dec_dim = dec_dim
        self.dec_layers = dec_layers
        self.dec_heads = dec_heads
        self.dec_mlp_units = dec_mlp_units

        self.output_projection = output_projection

        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.activation = activation

        self.final_shape = final_shape

        # computing number of patches
        if input_shape[0] % sinogram_height != 0 or input_shape[1] % sinogram_width != 0:
            raise ValueError("Cannot divide image into even patches")

        num_patches = int(input_shape[1] / sinogram_width * input_shape[0] / sinogram_height)

        # patch encoder
        self.patches = Patches(sinogram_width, sinogram_height, f'{name}_patches')
        self.patch_encoder = PatchEncoder(
            num_patches=num_patches,
            projection_dim=enc_dim,
            name=f'{name}_enc_projection'
        )

        self.patch_encoder_2 = PatchEncoder(
            num_patches=num_patches,
            projection_dim=dec_dim,
            name=f"{name}_dec_projection"
        )

        # building encoder layer
        self.encoder = [
            TransformerEncoder(
                project_dim=enc_dim,
                num_heads=enc_heads,
                mlp_dim=enc_mlp_units,
                mlp_dropout=dropout,
                attention_dropout=dropout,
                activation=activation,
                divide_heads=divide_heads,
                layer_norm_epsilon=layer_norm_epsilon,
                name=f"{name}_enc_block_{i}"
            ) for i in range(enc_layers)
        ]

        # building decoder layers
        self.decoder = [
            TransformerEncoder(
                project_dim=dec_dim,
                num_heads=dec_heads,
                mlp_dim=dec_mlp_units,
                mlp_dropout=dropout,
                attention_dropout=dropout,
                activation=activation,
                divide_heads=divide_heads,
                layer_norm_epsilon=layer_norm_epsilon,
                name=f"{name}_dec_block_{i}"
            ) for i in range(dec_layers)
        ]

        # final decoder layers
        self.output_projection = Sequential([
            LayerNormalization(epsilon=layer_norm_epsilon, name=f'{name}_output_projection_norm'),
            Dense(dec_dim, name=f'{name}_output_projection_dense'),
        ], name=f'{name}_output_projection')

        self.patch_decoder = PatchDecoder(
            output_patch_width,
            output_patch_height,
            output_x_patches,
            output_y_patches,
            name=f"{name}_depatchify"
        )

        # resize so the model will use the legit resolution and give back the legit PSNR
        self.resize = Resizing(
            final_shape[0], final_shape[1]
        )

    def call(self, inputs, training=None, mask=None):
        x = self.patches(inputs)
        x = self.patch_encoder(x)

        # encoder
        for block in self.encoder:
            x = block(x)

        # decoder projection
        if self.dec_projection:
            x = self.patch_encoder_2(x)

        # decoder
        for block in self.decoder:
            x = block(x)

        # output projection
        if self.output_projection:
            x = self.output_projection(x)

        # reshape
        return self.resize(self.patch_decoder(x))


if __name__ == "__main__":
    import os

    os.environ["KERAS_BACKEND"] = "tensorflow"

    import keras_core

    model = DirectReconstructionModel(
        input_shape=(1024, 513, 1),
        final_shape=(362, 362, 1),
        enc_layers=4,
        dec_layers=4,
        sinogram_width=513,
        sinogram_height=1,
        enc_dim=512,
        enc_mlp_units=2048,
        dec_dim=256,
        dec_mlp_units=2048,
        output_patch_width=16,
        output_patch_height=16,
        output_x_patches=32,
        output_y_patches=32,
        output_projection=True
    )
    model.call(keras_core.random.normal(shape=(1, 1024, 513, 1)))

    model.compile(optimizer=keras_core.optimizers.AdamW(learning_rate=5e-5, weight_decay=1e-5), loss='mse')
    model.summary()

    model.save_weights("model.weights.h5")
    model.save("model.keras")
