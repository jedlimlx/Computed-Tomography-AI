import keras_core as kc
import keras_core.ops as ops

import tensorflow as tf

from keras_core.layers import *
from keras_nlp.layers import PositionEmbedding, SinePositionEncoding


class Patches(Layer):
    def __init__(self, patch_width, patch_height, name=None, **kwargs):
        super(Patches, self).__init__(name=name, **kwargs)
        self.patch_width = patch_width
        self.patch_height = patch_height

    def call(self, images, **kwargs):
        batch_size = ops.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_height, self.patch_width, 1],
            strides=[1, self.patch_height, self.patch_width, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = ops.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        cfg = super(Patches, self).get_config()
        cfg.update({
            'patch_width': self.patch_width,
            'patch_height': self.patch_height
        })

        return cfg


class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, embedding_type='sin_cos', name=None, **kwargs):
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


class PatchDecoder(Layer):
    def __init__(self, patch_width, patch_height, x_patches, y_patches, channels=1, ignore_last=False, name=None,
                 **kwargs):
        super(PatchDecoder, self).__init__(name=name, **kwargs)
        self.channels = channels
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.x_patches = x_patches
        self.y_patches = y_patches
        self.ignore_last = ignore_last

    def call(self, encoded, **kwargs):
        # extracting patches
        if self.ignore_last: encoded = encoded[:, :-1, :]
        reshaped = ops.reshape(encoded, (-1, self.y_patches, self.x_patches, self.patch_height, self.patch_width))
        reshaped = ops.transpose(reshaped, [0, 1, 3, 2, 4])
        reshaped = ops.reshape(reshaped, (-1, self.y_patches * self.patch_height, self.x_patches * self.patch_width, self.channels))

        return reshaped

    def get_config(self):
        cfg = super(PatchDecoder, self).get_config()
        cfg.update({
            'patch_width': self.patch_width,
            'patch_height': self.patch_height,
            'x_patches': self.x_patches,
            'y_patches': self.y_patches
        })
        return cfg


class TransformerEncoder(Layer):  # todo use keras cv implementation when it becomes better
    """
    Transformer encoder block implementation as a Keras Layer.

    Args:
        project_dim: the dimensionality of the projection of the encoder, and
            output of the `MultiHeadAttention`
        mlp_dim: the intermediate dimensionality of the MLP head before
            projecting to `project_dim`
        num_heads: the number of heads for the `MultiHeadAttention` layer
        mlp_dropout: default 0.1, the dropout rate to apply between the layers
            of the MLP head of the encoder
        attention_dropout: default 0.1, the dropout rate to apply in the
            MultiHeadAttention layer
        activation: default 'tf.activations.gelu', the activation function to
            apply in the MLP head - should be a function
        layer_norm_epsilon: default 1e-06, the epsilon for `LayerNormalization`
            layers
    """

    def __init__(
            self,
            project_dim,
            num_heads,
            mlp_dim,
            mlp_dropout=0.1,
            attention_dropout=0.1,
            activation=kc.activations.gelu,
            layer_norm_epsilon=1e-06,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.mlp_dropout = mlp_dropout
        self.attention_dropout = attention_dropout
        self.activation = Activation(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.mlp_units = [mlp_dim, project_dim]

        self.layer_norm1 = LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )
        self.layer_norm2 = LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )
        self.attn = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.project_dim // self.num_heads,
            dropout=self.attention_dropout,
        )
        self.dense1 = Dense(self.mlp_units[0])
        self.dense2 = Dense(self.mlp_units[1])

        self.dropout = Dropout(self.mlp_dropout)

    def call(self, inputs):
        """Calls the Transformer Encoder on an input sequence.
        Args:
            inputs: A `tf.Tensor` of shape [batch, height, width, channels]

        Returns:
            `A tf.Tensor` of shape [batch, patch_num+1, embedding_dim]
        """

        if inputs.shape[-1] != self.project_dim:
            raise ValueError(
                "The input and output dimensionality must be the same, but the "
                f"TransformerEncoder was provided with {inputs.shape[-1]} and "
                f"{self.project_dim}"
            )

        x = self.layer_norm1(inputs)
        x = self.attn(x, x)
        x = self.dropout(x)
        x = x + inputs

        y = self.layer_norm2(x)

        y = self.dense1(y)
        if self.activation == kc.activations.gelu:
            y = self.activation(y, approximate=True)
        else:
            y = self.activation(y)
        y = self.dropout(y)
        y = self.dense2(y)
        y = self.dropout(y)

        output = x + y

        return output

    def get_config(self):
        config = super().get_config()
        activation = self.activation
        if not isinstance(activation, (str, dict)):
            activation = kc.activations.serialize(activation)
        config.update(
            {
                "project_dim": self.project_dim,
                "mlp_dim": self.mlp_dim,
                "num_heads": self.num_heads,
                "attention_dropout": self.attention_dropout,
                "mlp_dropout": self.mlp_dropout,
                "activation": activation,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        activation = config.pop("activation")
        if isinstance(activation, (str, dict)):
            activation = kc.activations.deserialize(activation)
        return cls(activation=activation, **config)


class TransformerDecoder(Layer):  # todo use keras cv implementation when it becomes better
    """
    Transformer decoder block implementation as a Keras Layer.

    Args:
        project_dim: the dimensionality of the projection of the encoder, and
            output of the `MultiHeadAttention`
        mlp_dim: the intermediate dimensionality of the MLP head before
            projecting to `project_dim`
        num_heads: the number of heads for the `MultiHeadAttention` layer
        mlp_dropout: default 0.1, the dropout rate to apply between the layers
            of the MLP head of the encoder
        attention_dropout: default 0.1, the dropout rate to apply in the
            MultiHeadAttention layer
        activation: default 'tf.activations.gelu', the activation function to
            apply in the MLP head - should be a function
        layer_norm_epsilon: default 1e-06, the epsilon for `LayerNormalization`
            layers
    """

    def __init__(
            self,
            project_dim,
            num_heads,
            enc_dim,
            mlp_dim,
            mlp_dropout=0.1,
            attention_dropout=0.1,
            activation=kc.activations.gelu,
            layer_norm_epsilon=1e-06,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.mlp_dim = mlp_dim
        self.enc_dim = enc_dim
        self.num_heads = num_heads
        self.mlp_dropout = mlp_dropout
        self.attention_dropout = attention_dropout
        self.activation = Activation(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.mlp_units = [mlp_dim, project_dim]

        # layer norms
        self.layer_norm1 = LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )
        self.layer_norm2 = LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )
        self.layer_norm3 = LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )

        # attention
        self.attn = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.project_dim // self.num_heads,
            dropout=self.attention_dropout
        )

        self.cross_attn = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.enc_dim // self.num_heads,
            dropout=self.attention_dropout
        )

        # MLP projection
        self.dense1 = Dense(self.mlp_units[0])
        self.dense2 = Dense(self.mlp_units[1])

        self.dropout = Dropout(self.mlp_dropout)

    def call(self, inputs):
        encoder_inputs, decoder_inputs = inputs

        if encoder_inputs.shape[-1] != self.project_dim:
            raise ValueError(
                "The input and output dimensionality must be the same, but the "
                f"TransformerDecoder was provided with {inputs.shape[-1]} and "
                f"{self.project_dim}"
            )

        # self-attention part
        x = self.layer_norm1(encoder_inputs)
        x = self.attn(x, x)
        x = self.dropout(x)
        x = x + inputs

        y = self.layer_norm2(x)
        y = self.attn(y, decoder_inputs)
        y = self.dropout(y)
        y = x + y

        z = self.layer_norm3(y)
        z = self.dense1(z)
        if self.activation == kc.activations.gelu:
            z = self.activation(z, approximate=True)
        else:
            z = self.activation(z)

        z = self.dropout(z)
        z = self.dense2(z)
        z = self.dropout(z)

        output = y + z
        return output

    def get_config(self):
        config = super().get_config()
        activation = self.activation
        if not isinstance(activation, (str, dict)):
            activation = kc.activations.serialize(activation)
        config.update(
            {
                "project_dim": self.project_dim,
                "mlp_dim": self.mlp_dim,
                "enc_dim": self.enc_dim,
                "num_heads": self.num_heads,
                "attention_dropout": self.attention_dropout,
                "mlp_dropout": self.mlp_dropout,
                "activation": activation,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        activation = config.pop("activation")
        if isinstance(activation, (str, dict)):
            activation = kc.activations.deserialize(activation)
        return cls(activation=activation, **config)
