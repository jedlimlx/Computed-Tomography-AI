"""
Fixes some bugs in the keras implementation of `map_coordinates`
"""
import functools
import itertools
import operator

import tensorflow as tf
from keras.ops import convert_to_tensor


def _mirror_index_fixer(index, size):
    s = size - 1  # Half-wavelength of triangular wave
    # Scaled, integer-valued version of the triangular wave |x - round(x)|
    return tf.abs((index + s) % (2 * s) - s)


def _reflect_index_fixer(index, size):
    return tf.math.floordiv(
        _mirror_index_fixer(2 * index + 1, 2 * size + 1) - 1, 2
    )


_INDEX_FIXERS = {
    "constant": lambda index, size: index,
    "nearest": lambda index, size: tf.clip_by_value(index, 0, size - 1),
    "wrap": lambda index, size: index % size,
    "mirror": _mirror_index_fixer,
    "reflect": _reflect_index_fixer,
}


def _nearest_indices_and_weights(coordinate):
    coordinate = (
        coordinate if coordinate.dtype.is_integer else tf.round(coordinate)
    )
    index = tf.cast(coordinate, tf.int32)
    weight = tf.constant(1, coordinate.dtype)
    return [(index, weight)]


def _linear_indices_and_weights(coordinate):
    lower = tf.floor(coordinate)
    upper_weight = coordinate - lower
    lower_weight = 1 - upper_weight
    index = tf.cast(lower, tf.int32)
    return [(index, lower_weight), (index + 1, upper_weight)]


def map_coordinates(
    input, coordinates, order, fill_mode="constant", fill_value=0.0
):
    input_arr = convert_to_tensor(input)
    coordinate_arrs = convert_to_tensor(coordinates)
    # unstack into a list of tensors for following operations
    coordinate_arrs = tf.unstack(coordinate_arrs, axis=0)
    fill_value = convert_to_tensor(tf.cast(fill_value, input_arr.dtype))

    if coordinates.shape[0] != len(input_arr.shape):
        raise ValueError(
            "coordinates must be a sequence of length input.shape, but "
            f"{len(coordinates)} != {len(input_arr.shape)}"
        )

    index_fixer = _INDEX_FIXERS.get(fill_mode)
    if index_fixer is None:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected one of "
            f"{set(_INDEX_FIXERS.keys())}. Received: "
            f"fill_mode={fill_mode}"
        )

    def is_valid(index, size):
        if fill_mode == "constant":
            return (0 <= index) & (index < size)
        else:
            return True

    if order == 0:
        interp_fun = _nearest_indices_and_weights
    elif order == 1:
        interp_fun = _linear_indices_and_weights
    else:
        raise NotImplementedError("map_coordinates currently requires order<=1")

    valid_1d_interpolations = []
    for coordinate, size in zip(coordinate_arrs, input_arr.shape):
        interp_nodes = interp_fun(coordinate)
        valid_interp = []
        for index, weight in interp_nodes:
            fixed_index = index_fixer(index, size)
            valid = is_valid(index, size)
            valid_interp.append((fixed_index, valid, weight))
        valid_1d_interpolations.append(valid_interp)

    outputs = []
    for items in itertools.product(*valid_1d_interpolations):
        indices, validities, weights = zip(*items)
        indices = tf.transpose(tf.stack(indices))

        def fast_path():
            return tf.transpose(tf.gather_nd(input_arr, indices))

        def slow_path():
            all_valid = functools.reduce(operator.and_, validities)
            return tf.where(
                all_valid,
                tf.transpose(tf.gather_nd(input_arr, indices)),
                fill_value,
            )

        contribution = tf.cond(tf.reduce_all(validities), fast_path, slow_path)
        outputs.append(
            functools.reduce(operator.mul, weights)
            * tf.cast(contribution, weights[0].dtype)
        )
    result = functools.reduce(operator.add, outputs)
    if input_arr.dtype.is_integer:
        result = result if result.dtype.is_integer else tf.round(result)
    return tf.cast(result, input_arr.dtype)