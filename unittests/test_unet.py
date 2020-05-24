import pytest
import tensorflow as tf

from unet.model_tf import Unet


@pytest.fixture
def unet():
    model = Unet(num_classes=2, net_depth=3)
    return model

@pytest.fixture
def basic_image():
    return tf.ones((2, 128, 128, 1))


def test_unet_size(unet, basic_image):
    prediction = unet(basic_image)
    assert prediction.get_shape() == (2, 128, 128, 2)


def test_additional_depth(basic_image):
    model = Unet(num_classes=10, net_depth=5)
    prediction = model(basic_image)
    assert prediction.get_shape() == (2, 128, 128, 10)

def test_unet_gradients(basic_image):
    labels = basic_image
    model = Unet(num_classes=1, net_depth=1)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        predictions = model(basic_image)
        loss = loss_fn(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    assert all([g is not None for g in gradients])

