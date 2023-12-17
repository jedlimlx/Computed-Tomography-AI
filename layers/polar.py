from keras import ops
from keras.layers import *
import numpy as np
from utils.interpolate_bilinear import interpolate_bilinear


class PolarTransformBase(Layer):
    def __init__(self, out_shape, center=None, max_radius=None, order=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_shape = out_shape
        self.order = order
        self.center = center
        self.max_radius = max_radius
        self.input_shape = None
        self.coordinates = None

    def call(self, inputs, *args, **kwargs):
        img = inputs

        out = interpolate_bilinear(img, self.coordinates, indexing='ij')
        out = ops.reshape(out, (-1, *self.out_shape, img.shape[-1]))
        return out


class PolarTransform(PolarTransformBase):
    def build(self, input_shape):
        super().build(input_shape)
        self.input_shape = input_shape
        center = self.center or (input_shape[1] / 2, input_shape[2] / 2)
        max_radius = self.max_radius or np.sqrt(input_shape[1] ** 2 + input_shape[2] ** 2) / 2
        k_theta = self.out_shape[0] / (2 * np.pi)
        k_r = self.out_shape[1] / max_radius

        theta = ops.arange(self.out_shape[0], dtype=self.dtype) / k_theta
        theta = ops.reshape(theta, (-1, 1))
        r = ops.arange(self.out_shape[1], dtype=self.dtype) / k_r
        r = ops.reshape(r, (1, -1))

        coord_1 = r * ops.cos(theta) + center[0]
        coord_2 = r * ops.sin(theta) + center[1]

        coord_1 = ops.clip(coord_1, 0, input_shape[1] - 1)
        coord_2 = ops.clip(coord_2, 0, input_shape[2] - 1)
        self.coordinates = ops.stack([coord_1, coord_2], axis=-1)
        self.coordinates = ops.reshape(self.coordinates, (1, -1, 2))


class InvPolarTransform(PolarTransformBase):
    def build(self, input_shape):
        super().build(input_shape)
        self.input_shape = input_shape
        center = self.center or (self.out_shape[0] / 2, self.out_shape[1] / 2)
        max_radius = self.max_radius or np.sqrt(self.out_shape[0] ** 2 + self.out_shape[1] ** 2) / 2

        k_theta = self.input_shape[1] / (2 * np.pi)
        k_r = self.input_shape[2] / max_radius

        x = ops.arange(self.out_shape[0], dtype=self.dtype) - center[0]
        x = ops.reshape(x, (-1, 1))
        y = ops.arange(self.out_shape[1], dtype=self.dtype) - center[1]
        y = ops.reshape(y, (1, -1))

        coord_1 = k_theta * ops.mod(ops.arctan2(y, x), 2 * np.pi)
        coord_2 = k_r * ops.sqrt(x ** 2 + y ** 2)

        coord_1 = ops.clip(coord_1, 0, input_shape[1] - 1)
        coord_2 = ops.clip(coord_2, 0, input_shape[2] - 1)
        self.coordinates = ops.stack([coord_1, coord_2], axis=-1)
        self.coordinates = ops.reshape(self.coordinates, (1, -1, 2))
