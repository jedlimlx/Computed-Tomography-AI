from keras_core.layers import *
from keras_core.models import *
from keras_nlp.layers import SinePositionEncoding, PositionEmbedding

from models.masked_sinogram_autoencoder import Patches, SinogramPatchEncoder, PatchDecoder, TransformerEncoder


class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, embedding_type='learned', name=None, **kwargs):
        """
        Projection and embedding
        Args:
            num_patches: Sequence length
            projection_dim: Output dim after projection
            embedding_type: Type of embedding used, 'sin_cos' or 'learned'
            name: Name for this op
        """
        assert embedding_type in ['sin_cos', 'learned']  # error checking
        super(PatchEncoder, self).__init__(name=name, **kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.embedding_type = embedding_type

        # projection
        self.projection = Dense(units=projection_dim)

        # positional encoding
        if embedding_type == 'learned':
            self.position_embedding = PositionEmbedding(num_patches)
        else:
            self.position_embedding = SinePositionEncoding(num_patches)

    def call(self, patch, **kwargs):
        encoded = self.projection(patch)
        embedding = self.position_embedding(encoded)

        return encoded + embedding

    def get_config(self):
        cfg = super(PatchEncoder, self).get_config()
        cfg.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,
            'embedding_type': self.embedding_type
        })
        return cfg


class DirectReconstructionModel(Model):
    def __init__(
            self,
            input_shape=(256, 256, 1),
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
            activation='gelu',
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
        self.activation = activation

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
                name=f"{name}_dec_block_{i}"
            ) for i in range(dec_layers)
        ]

        # final decoder layers
        self.output_projection = Dense(
            sinogram_width * sinogram_height,
            name=f'{name}_output_projection'
        )

        self.patch_decoder = PatchDecoder(
            output_patch_width,
            output_patch_height,
            output_x_patches,
            output_y_patches,
            name=f"{name}_depatchify"
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
        return self.patch_decoder(x)


if __name__ == "__main__":
    import os

    os.environ["KERAS_BACKEND"] = "tensorflow"

    import keras_core

    model = DirectReconstructionModel(
        input_shape=(256, 256, 1),
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
        activation='gelu'
    )
    model.call(keras_core.random.normal(shape=(1, 256, 256, 1)))

    model.compile(optimizer=keras_core.optimizers.AdamW(learning_rate=5e-5, weight_decay=1e-5), loss='mse')
    model.summary()

    model.save_weights("model.weights.h5")
    model.save("model.keras")
