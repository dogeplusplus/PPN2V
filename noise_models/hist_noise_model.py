############################################
#    The Noise Model
############################################

import numpy as np
import tensorflow as tf


def create_histogram(bins, min_val, max_val, observation, signal):
    """
    Creates a 2D histogram from 'observation' and 'signal'

    Parameters
    ----------
    bins: int
        The number of bins in x and y. The total number of 2D bins is 'bins'**2.
    minVal: float
        the lower bound of the lowest bin in x and y.
    maxVal: float
        the highest bound of the highest bin in x and y.
    observation: numpy array
        A 3D numpy array that is interpretted as a stack of 2D images.
        The number of images has to be divisible by the number of images in 'signal'.
        It is assumed that n subsequent images in observation belong to one image image in 'signal'.
    signal: numpy array
        A 3D numpy array that is interpretted as a stack of 2D images.

    Returns
    ----------
    histogram: numpy array
        A 3D array:
        'histogram[0,...]' holds the normalized 2D counts.
        Each row sums to 1, describing p(x_i|s_i).
        'histogram[1,...]' holds the lower boundaries of each bin in y.
        'histogram[2,...]' holds the upper boundaries of each bin in y.
        The values for x can be obtained by transposing 'histogram[1,...]' and 'histogram[2,...]'.
    """

    imgFactor = int(observation.shape[0] / signal.shape[0])
    histogram = np.zeros((3, bins, bins))
    ra = [min_val, max_val]

    for i in range(observation.shape[0]):
        observation_ = observation[i].copy().ravel()

        signal_ = (signal[i // imgFactor].copy()).ravel()

        a = np.histogram2d(signal_, observation_, bins=bins, range=[ra, ra])
        histogram[0] = histogram[0] + a[0] + 1e-30  # This is for numerical stability

    for i in range(bins):
        if (
                np.sum(histogram[0, i, :]) > 1e-20
        ):  # We exclude empty rows from normalization
            histogram[0, i, :] /= np.sum(
                histogram[0, i, :]
            )  # we normalize each non-empty row

    for i in range(bins):
        histogram[1, :, i] = a[1][
                             :-1
                             ]  # The lower boundaries of each bin in y are stored in dimension 1
        histogram[2, :, i] = a[1][
                             1:
                             ]  # The upper boundaries of each bin in y are stored in dimension 2
        # The accordent numbers for x are just transopsed.

    return histogram


class NoiseModel:
    def __init__(self, histogram):
        """
        Creates a NoiseModel object.

        Parameters
        ----------
        histogram: numpy array
            A histogram as create by the 'createHistogram(...)' method.
        """

        # The number of bins is the same in x and y
        bins = histogram.shape[1]

        # The lower boundaries of each bin in y are stored in dimension 1
        self.minv = np.min(histogram[1, ...])

        # The upper boundaries of each bin in y are stored in dimension 2
        self.maxv = np.max(histogram[2, ...])

        # move everything to GPU
        self.bins = bins
        self.full_hist = tf.constant(histogram[0, ...].astype(np.float32))

    def likelihood(self, obs, signal):
        """
        Calculate the likelihood p(x_i|s_i) for every pixel in a tensor, using a histogram based noise model.
        To ensure differentiability in the direction of s_i, we linearly interpolate in this direction.

        Parameters
        ----------
        obs: tensor
            tensor holding your observed intensities x_i.

        signal: tensor
            tensor holding hypotheses for the clean signal at every pixel s_i^k.

        Returns
        ----------
        Torch tensor containing the observation likelihoods according to the noise model.
        """
        # print(obs.get_shape())
        # print(signal.get_shape())
        obsF = self.get_index_obs_float(obs)
        obs_ = tf.cast(tf.math.floor(obsF), tf.int32)
        signalF = self.get_index_signal_float(signal)
        signal_ = tf.cast(tf.math.floor(signalF), tf.int32)
        fact = signalF - tf.cast(signal_, tf.float32)

        # Finally we are looking up the values and interpolate
        # TODO: this seems to work but is ugly and quite slow
        # TODO: [4,120, 120,1] vs [4, 120, 120] add operation fails here when training
        # TODO: rethink how the likelihood is calculated for each sample.
        # The array should not be reduced, the reduce_mean should happen in the loss calculation
        # print(signal_.get_shape())
        # print(obs_.get_shape())
        raw_indices = tf.broadcast_to(256 * signal_, obs_.get_shape()) + obs_
        converted_indices = tf.reshape(
            tf.unravel_index(
                tf.reshape(raw_indices, [-1]), dims=self.full_hist.get_shape()

            ),
            raw_indices.get_shape() + [2],
        )
        x0 = tf.gather_nd(self.full_hist, converted_indices)

        second_indices = tf.broadcast_to(tf.clip_by_value(signal_ + 1, 0, self.bins), obs_.get_shape()) + obs_
        converted_indices_2 = tf.reshape(
            tf.unravel_index(
                tf.reshape(second_indices, [-1]), dims=self.full_hist.get_shape()
            ),
            second_indices.get_shape() + [2],
        )
        x1 = tf.gather_nd(self.full_hist, converted_indices_2)
        a = x0 * (1.0 - tf.broadcast_to(fact, x0.get_shape()))
        b = x1 * tf.broadcast_to(fact, x1.get_shape())
        return a + b

    def get_index_obs_float(self, x):
        return tf.clip_by_value(
            self.bins * (x - self.minv) / (self.maxv - self.minv),
            clip_value_min=0.0,
            clip_value_max=self.bins - 1 - 1e-3,
        )

    def get_index_signal_float(self, x):
        return tf.clip_by_value(
            self.bins * (x - self.minv) / (self.maxv - self.minv),
            clip_value_min=0.0,
            clip_value_max=self.bins - 1 - 1e-3,
        )
