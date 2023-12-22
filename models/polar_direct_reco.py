import tensorflow as tf
from keras.models import *
from keras import ops

from layers import TransformerEncoder, PolarTransform, InvPolarTransform


class PolarDirectReco(Model):
    def __init__(
            self,
            sinogram_encoder,
            out_shape,
            dec_blocks=4,
            dec_dim=512,
            dec_heads=16,
            dropout=0.,
            layer_norm_epsilon=1e-5,
            ipt_when_training=False,
            ipt_when_testing=True,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dec_dim = dec_dim
        self.out_shape = out_shape
        self.sinogram_encoder = sinogram_encoder
        self.ipt_when_training = ipt_when_training
        self.ipt_when_testing = ipt_when_testing
        self.polar_transform = PolarTransform(
            (sinogram_encoder.inp_shape[0], dec_dim),
            name=f'{self.name}-polar_transform'
        )
        self.inv_polar_transform = InvPolarTransform(
            out_shape,
            name=f'{self.name}-inverse_polar_transform'
        )

        self.decoder_blocks = [
            TransformerEncoder(
                project_dim=dec_dim,
                num_heads=dec_heads,
                mlp_dim=dec_dim * 4,
                mlp_dropout=dropout,
                attention_dropout=dropout,
                activation='gelu',
                layer_norm_epsilon=layer_norm_epsilon,
                name=f'{self.name}-decoder_block_{i}'
            )
            for i in range(dec_blocks)
        ]

    def build(self, input_shape):
        if not self.polar_transform.built:
            self.polar_transform.build(input_shape)
        if not self.inv_polar_transform.built:
            self.inv_polar_transform.build((self.out_shape, self.sinogram_encoder.inp_shape[0], self.dec_dim))
        self.built = True

    def call(self, x, *args, **kwargs):
        x = self.sinogram_encoder.patches(x)
        x, _, _ = self.sinogram_encoder.encoder_impl(x)

        y = x
        for block in self.decoder_blocks:
            y = block(y)

        return ops.expand_dims(y, -1)

    def train_step(self, data):
        sinogram, gt = data

        if not self.ipt_when_training:
            gt = self.polar_transform(gt)

        with tf.GradientTape() as tape:
            output = self(sinogram)
            if self.ipt_when_training:
                output = self.inv_polar_transform(output)

            loss = self.compute_loss(y=gt, y_pred=output)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(gt, output)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        sinogram, gt = data

        if not self.ipt_when_testing:
            gt = self.polar_transform(gt)

        output = self(sinogram)
        if self.ipt_when_testing:
            output = self.inv_polar_transform(output)

        loss = self.compute_loss(y=gt, y_pred=output)

        # Update the metrics.
        # Metrics are configured in `compile()`.
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(gt, output)

        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        output = self(data)
        return self.inv_polar_transform(output)


if __name__ == '__main__':
    def test():
        from models.masked_sinogram_autoencoder import MaskedSinogramAutoencoder
        from keras import optimizers

        def transform(_):
            x = tf.random.normal(shape=(2, 1024, 513, 1))
            y = tf.random.normal(shape=(2, 362, 362, 1))
            return x, y

        train_ds = tf.data.Dataset.random(seed=1).map(transform)
        val_ds = tf.data.Dataset.random(seed=1).map(transform)

        autoencoder = MaskedSinogramAutoencoder(
            enc_layers=4,
            dec_layers=1,
            sinogram_width=513,
            sinogram_height=1,
            input_shape=(1024, 513, 1),
            enc_dim=1024,
            enc_mlp_units=2048,
            dec_dim=1024,
            dec_mlp_units=2048,
            mask_ratio=0.75
        )
        model = PolarDirectReco(
            autoencoder,
            out_shape=(362, 362),
            dec_blocks=1,
            dec_dim=1024,
            dec_heads=16,
            dropout=0.,
            layer_norm_epsilon=1e-5,
            ipt_when_training=False,
            ipt_when_testing=True,
        )
        model.compile(loss='mse', optimizer=optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5))
        model.build(((None, 1024, 513, 1), (None, 512, 512, 1)))
        model.fit(train_ds, validation_data=val_ds, steps_per_epoch=1, validation_steps=1)

        print(model.summary())


    test()

