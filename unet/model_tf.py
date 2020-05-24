import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Cropping2D, Concatenate, UpSampling2D

from typing import NoReturn, Tuple


class Unet(Model):
    def __init__(self, num_classes: int, net_depth: int, initial_filters: int = 64):
        super(Unet, self).__init__()
        self._build_model(num_classes, net_depth, initial_filters)
        self.std = tf.math.log(0.5)

    def _build_model(self, num_classes: int, net_depth: int, filters: int) -> NoReturn:
        self.downsampling_layers = []

        for i in range(net_depth):
            self.downsampling_layers.append([
                Conv2D(filters * 2 ** i, kernel_size=3, activation='relu', padding='same') for _ in range(2)
            ])

        self.bottom_convolutions = [
            Conv2D(filters * 2 ** (net_depth), kernel_size=3, activation='relu', padding='same'),
            Conv2D(filters * 2 ** (net_depth - 1), kernel_size=3, activation='relu', padding='same')
        ]

        self.upsampling_layers = []
        for i in range(net_depth):
            self.upsampling_layers.append([
                Conv2D(filters * 2 ** (net_depth - i - 1), kernel_size=3, activation='relu', padding='same') for _ in
                range(2)
            ])

        self.last = Conv2D(num_classes, kernel_size=1)

    def call(self, inputs, training: bool = None, mask=None):
        x = inputs
        concatenating_pairs = []
        for downsample in self.downsampling_layers:
            for layer in downsample:
                x = layer(x)
            concatenating_pairs.insert(0, x)
            x = MaxPooling2D((2, 2))(x)

        for layer in self.bottom_convolutions:
            x = layer(x)

        for i, upsample in enumerate(self.upsampling_layers):
            x = UpSampling2D((2, 2))(x)
            previous = concatenating_pairs[i]
            cropping_dims = get_cropping_dims(previous, x)
            cropped = Cropping2D(cropping_dims)(previous)
            x = Concatenate()([cropped, x])
            for layer in upsample:
                x = layer(x)

        y = self.last(x)
        return y


def get_cropping_dims(previous: tf.Tensor, upsample: tf.Tensor) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Get cropping dimensions for the U-Net during the upsampling

    Args:
        previous: model output before downsampling
        upsample: pair model output after upsampling

    Returns:
        Cropping dimensions for the previous tensor
    """
    get_shape = lambda x: np.array(x.get_shape().as_list())
    previous_shape = get_shape(previous)
    upsample_shape = get_shape(upsample)
    padding = (previous_shape - upsample_shape)[1:3]
    cropping_dims = tuple([(int(x // 2), (x // 2)) for x in padding])
    return cropping_dims
