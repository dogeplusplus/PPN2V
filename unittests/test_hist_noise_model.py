import pytest
import numpy as np

from noise_models.hist_noise_model import create_histogram, NoiseModel


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
    assert histogram.shape == (3, 32, 32)


def test_likelihood(noise_model):
    observation = np.ones((18, 64, 64), np.float32)
    signal = np.ones((18, 64, 64), np.float32)

    likely = noise_model.likelihood(observation, signal)
    assert likely.shape == observation.shape

def test_likelihood_extra_dim(noise_model):
    observation = np.ones((18, 64, 64, 1), np.float32)
    signal = np.ones((18, 64, 64, 1), np.float32)

    likely = noise_model.likelihood(observation, signal)
    assert likely.shape == observation.shape
