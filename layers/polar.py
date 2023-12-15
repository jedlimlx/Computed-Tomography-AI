from keras import ops
from keras.layers import *
import numpy as np


class PolarTransformBase(Layer):
    def __init__(self, out_shape, center=None, max_radius=None, order=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_shape = out_shape
        self.order = order
        self.center = center
        self.max_radius = max_radius
        self.input_shape = None
        self.coord_1 = None
        self.coord_2 = None

    def call(self, inputs, *args, **kwargs):
        img = ops.transpose(inputs, [0, 3, 1, 2])
        img = ops.reshape(img, (-1, inputs.shape[1], inputs.shape[2]))
        coordinates = self._get_batch_coordinates_and_stack_coordinate_grid(ops.shape(img)[0], self.coord_1,
                                                                            self.coord_2)
        out = ops.image.map_coordinates(img, coordinates, order=self.order, fill_mode='nearest', fill_value=0)
        out = ops.reshape(out, (-1, inputs.shape[3], out.shape[1], out.shape[2]))
        out = ops.transpose(out, (0, 2, 3, 1))
        return out

    def _get_batch_coordinates_and_stack_coordinate_grid(self, batch_size, coord_1, coord_2):
        b = ops.arange(batch_size, dtype=self.dtype)
        b = ops.reshape(b, (-1, 1, 1))
        b = b * ops.ones_like(coord_1)
        x = coord_1 * ops.ones_like(b)
        y = coord_2 * ops.ones_like(b)
        coordinates = ops.stack([b, x, y], axis=0)
        return coordinates


class PolarTransform(PolarTransformBase):
    def build(self, input_shape):
        self.input_shape = input_shape
        center = self.center or (input_shape[1] / 2, input_shape[2] / 2)
        max_radius = self.max_radius or np.sqrt(input_shape[1] ** 2 + input_shape[2] ** 2) / 2
        k_theta = self.out_shape[0] / (2 * np.pi)
        k_r = self.out_shape[1] / max_radius

        theta = ops.arange(self.out_shape[0], dtype=self.dtype) / k_theta
        theta = ops.reshape(theta, (-1, 1))
        r = ops.arange(self.out_shape[1], dtype=self.dtype) / k_r
        r = ops.reshape(r, (1, -1))

        self.coord_1 = r * ops.cos(theta) + center[0]
        self.coord_2 = r * ops.sin(theta) + center[1]


class InvPolarTransform(PolarTransformBase):
    def build(self, input_shape):
        self.input_shape = input_shape
        center = self.center or (self.out_shape[0] / 2, self.out_shape[1] / 2)
        max_radius = self.max_radius or np.sqrt(self.out_shape[0] ** 2 + self.out_shape[1] ** 2) / 2

        k_theta = self.input_shape[1] / (2 * np.pi)
        k_r = self.input_shape[2] / max_radius

        x = ops.arange(self.out_shape[0], dtype=self.dtype) - center[0]
        x = ops.reshape(x, (-1, 1))
        y = ops.arange(self.out_shape[1], dtype=self.dtype) - center[1]
        y = ops.reshape(y, (1, -1))

        self.coord_1 = k_theta * ops.mod(ops.arctan2(y, x), 2 * np.pi)
        self.coord_2 = k_r * ops.sqrt(x ** 2 + y ** 2)
