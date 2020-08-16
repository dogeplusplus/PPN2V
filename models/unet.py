from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, UpSampling2D

from typing import NoReturn


class Unet(Model):
    def __init__(self, num_classes: int, depth: int, initial_filters: int = 64):
        super(Unet, self).__init__()
        self._build_model(num_classes, depth, initial_filters)

    def _build_model(self, num_classes: int, depth: int, filters: int) -> NoReturn:
        self.downsampling_layers = []

        for i in range(depth):
            self.downsampling_layers.append([
                Conv2D(filters * 2 ** i, kernel_size=3, activation='relu', padding='same') for _ in range(2)
            ])

        self.bottom_convolutions = [
            Conv2D(filters * 2 ** (depth), kernel_size=3, activation='relu', padding='same'),
            Conv2D(filters * 2 ** (depth - 1), kernel_size=3, activation='relu', padding='same')
        ]

        self.upsampling_layers = []
        for i in range(depth):
            self.upsampling_layers.append([
                Conv2D(filters * 2 ** (depth - i - 1), kernel_size=3, activation='relu', padding='same') for _ in
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
            x = Concatenate()([previous, x])
            for layer in upsample:
                x = layer(x)

        y = self.last(x)
        return y

