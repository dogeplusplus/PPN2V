import pytest
import numpy as np

from pn2v_tf.hist_noise_model import create_histogram, NoiseModel


@pytest.fixture
def histogram():
    observation = np.ones((64, 64, 1))
    signal = observation
    prob_histogram = create_histogram(bins=32, min_val=0, max_val=100, observation=observation, signal=signal)
    return prob_histogram


@pytest.fixture
def noise_model(histogram):
    return NoiseModel(histogram)


def test_create_histogram(histogram):
    assert histogram.shape == (32, 32)


def test_likelihood(noise_model):
    observation = np.ones((18, 64, 64))
    signal = np.ones((18, 64, 64))

    likely = noise_model.likelihood(observation, signal)
    assert likely.shape == observation.shape
