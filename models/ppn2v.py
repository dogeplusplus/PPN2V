import numpy as np
import tensorflow as tf

from bunch import Bunch
from typing import NoReturn
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from models.unet import Unet
from noise_models.hist_noise_model import NoiseModel
from noise_models.training import random_crop_fri, image2tensor


class PPN2V(Model):
    def __init__(self, model_config: Bunch, mean: float, std: float) -> NoReturn:
        super(PPN2V, self).__init__()
        self._build_model(model_config)
        self.mean = mean
        self.std = std

    def _build_model(self, model_config: Bunch) -> NoReturn:
        assert model_config.type in ("care", "n2v", "pn2v"), "Model type must be either care, n2v, or pn2v"
        self.model_type = model_config.type
        noise_histogram = np.load(model_config.noise_model_path)
        self.noise_model = NoiseModel(noise_histogram)
        self.unet = Unet(model_config.num_classes, model_config.depth, model_config.initial_filters)
        self.optimizer = Adam()
        self.loss_fn = self.loss_pn2v
        if self.model_type in ("care", "n2v"):
            self.noise_fn = self.loss_n2v

    def preprocess_data(self, sample_data, data_counter, size, bs, num_pix, augment=True):
        # Init Variables
        inputs = []
        labels = []
        masks = []
        # Assemble mini batch
        for j in range(bs):
            im, l, m, data_counter = random_crop_fri(sample_data,
                                                     size,
                                                     num_pix,
                                                     counter=data_counter,
                                                     augment=augment,
                                                     supervised=self.model_type == "care")
            inputs.append(image2tensor(im))
            labels.append(image2tensor(l))
            masks.append(image2tensor(m))

        inputs_raw, labels, masks = tf.stack(inputs)[..., tf.newaxis], tf.stack(labels), tf.stack(masks)[
            ..., tf.newaxis]

        return inputs_raw, labels, masks

    def validate_step(self, inputs: tf.Tensor, labels: tf.Tensor, masks: tf.Tensor,
                      loss_tensor: tf.keras.metrics.Mean, training: bool = False) -> float:
        std = tf.constant(self.std)
        mean = tf.constant(self.mean)

        model_inputs = tf.convert_to_tensor(inputs - mean / std)
        # Forward step
        outputs = self.unet(model_inputs[..., tf.newaxis], training=training) * 10.0  # We found that this factor can speed up training

        samples = tf.transpose(outputs, (3, 0, 1, 2))
        # Denormalize
        samples = samples * std + mean

        loss = self.loss_fn(samples, labels, masks)
        loss_tensor.update_state(loss)
        return loss

    # TODO: add back in tf function decorators afterwards
    def train_step(self, inputs: tf.Tensor, labels: tf.Tensor, masks: tf.Tensor,
                   loss_tensor=tf.keras.metrics.Mean) -> NoReturn:
        with tf.GradientTape() as tape:
            loss = self.validate_step(inputs, labels, masks, loss_tensor, training=True)
        gradients = tape.gradient(loss, self.unet.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.unet.trainable_variables))

    def loss_pn2v(self, samples, labels, masks):
        '''
        The loss function as described in Eq. 7 of the paper.
        '''

        likelihoods = self.noise_model.likelihood(labels, samples)
        likelihoods_avg = tf.math.log(tf.reduce_mean(likelihoods, axis=0, keepdims=True)[0, ...])

        # Average over pixels and batch
        masks = tf.squeeze(masks)
        loss = -tf.reduce_sum(likelihoods_avg * masks) / tf.reduce_sum(masks)
        return loss

    def loss_n2v(self, samples, labels, masks):
        '''
        The loss function as described in Eq. 7 of the paper.
        '''

        errors = (labels - tf.reduce_mean(samples, axis=0)) ** 2

        # Average over pixels and batch
        loss = tf.reduce_sum(errors * masks) / tf.reduce_sum(masks)
        return loss / (self.std ** 2)

    def train(self, training_data: tf.data.Dataset, training_config: Bunch,
              validation_data: tf.data.Dataset = None) -> NoReturn:
        """
        Run training on the image

        Args:
            training_data: training image
            training_config: training parameters
            validation_data: optional validation image for testing
        """
        epochs = training_config.epochs
        steps_per_epoch = training_config.steps_per_epoch
        patch_size = training_config.patch_size
        virtual_batch_size = training_config.virtual_batch_size
        batch_size = training_config.batch_size

        losses = tf.keras.metrics.Mean(name='train_pn2v_loss')
        if validation_data is not None:
            val_losses = tf.keras.metrics.Mean(name='val_pn2v_loss')

        for epoch in range(epochs):
            for _ in range(steps_per_epoch * virtual_batch_size):
                for data in training_data:
                    images, labels, masks = data
                    self.train_step(images, labels, masks, losses)

                if validation_data is not None:
                    for data in validation_data:
                        val_images, val_labels, val_masks = data
                        val_loss = self.validate_step(val_images, val_labels, val_masks, val_losses)
                        val_losses.update_state(val_loss)

                losses.reset_states()
                if validation_data is not None:
                    val_losses.reset_states()
