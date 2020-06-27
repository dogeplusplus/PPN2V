import numpy as np

from utils.dataset import manipulate_pixels

def test_manipulate_pixels():
    random_image = np.array([i for i in range(100)]).reshape((10, 10))
    images, labels, masks = manipulate_pixels(random_image, 5)

    # Check the output shapes are the same
    assert random_image.shape == images.shape
    assert random_image.shape == masks.shape
    assert random_image.shape == labels.shape

    # Check that at least some of the input image has changed
    assert np.any(random_image != images)
