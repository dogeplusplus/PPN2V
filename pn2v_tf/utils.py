import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def image2tensor(img):
    """
    Convert a 2D single channel image to a pytorch tensor.
    """
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    return img_tensor


def fast_shuffle(series, num):
    length = series.shape[0]
    for i in range(num):
        series = series[np.random.permutation(length), :]
    return series


def plot_probability_distribution(
    signal_bin_index, histogram, gaussian_mixture_model, min_signal, max_signal, n_bin
):
    """Plots probability distribution P(x|s) for a certain ground truth signal.
       Predictions from both Histogram and GMM-based Noise models are displayed for comparison.
        Parameters
        ----------
        signal_bin_index: int
            index of signal bin. Values go from 0 to number of bins (`n_bin`).
        histogram: numpy array
            A square numpy array of size `nbin` times `n_bin`.
        gaussian_mixture_model: GaussianMixtureNoiseModel
            Object containing trained parameters.
        min_signal: float
            Lowest pixel intensity present in the actual sample which needs to be denoised.
        max_signal: float
            Highest pixel intensity present in the actual sample which needs to be denoised.
        n_bin: int
            Number of Bins.
        """
    hist_bin_size = (max_signal - min_signal) / n_bin
    querySignal_numpy = (
        signal_bin_index / float(n_bin) * (max_signal - min_signal) + min_signal
    )
    querySignal_numpy += hist_bin_size / 2
    querySignal_torch = np.array(querySignal_numpy, np.float32)

    query_observations = np.arange(min_signal, max_signal, hist_bin_size)
    query_observations += hist_bin_size / 2
    query_observations = np.array(query_observations, np.float32)
    pTorch = gaussian_mixture_model.likelihood(query_observations, querySignal_torch)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.xlabel("Observation Bin")
    plt.ylabel("Signal Bin")
    plt.imshow(histogram ** 0.25, cmap="gray")
    plt.axhline(y=signal_bin_index + 0.5, linewidth=5, color="blue", alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(
        query_observations,
        histogram[signal_bin_index, :] / hist_bin_size,
        label="GT Hist: bin =" + str(signal_bin_index),
        color="blue",
        linewidth=2,
    )
    plt.plot(
        query_observations,
        pTorch,
        label="GMM : " + " signal = " + str(np.round(querySignal_numpy, 2)),
        color="red",
        linewidth=2,
    )
    plt.xlabel("Observations (x) for signal s = " + str(querySignal_numpy))
    plt.ylabel("Probability Density")
    plt.title("Probability Distribution P(x|s) at signal =" + str(querySignal_numpy))
    plt.legend()
