import os
import tqdm
import logging
import numpy as np
import tensorflow as tf

from bunch import Bunch
from typing import NoReturn
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from models.unet import Unet
from noise_models.hist_noise_model import NoiseModel
from noise_models.training import random_crop_fri, image2tensor
from pn2v.utils import normalize_for_viz


class PPN2V(Model):
    def __init__(self, model_config: Bunch, mean: float, std: float) -> NoReturn:
        super(PPN2V, self).__init__()
        self._build_model(model_config)
        self.mean = mean
        self.std = std

    def _build_model(self, model_config: Bunch) -> NoReturn:
        assert model_config.type in (
            "care", "n2v", "pn2v"), "Model type must be either care, n2v, or pn2v"
        self.model_type = model_config.type
        noise_histogram = np.load(model_config.noise_model_path)
        self.noise_model = NoiseModel(noise_histogram)
        self.unet = Unet(model_config.num_classes, model_config.depth,
                         model_config.initial_filters)
        self.optimizer = Adam()
        self.loss_fn = self.loss_pn2v
        self.model_path = model_config.model_path
        os.makedirs(self.model_path)
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

        inputs_raw, labels, masks = tf.stack(inputs)[..., tf.newaxis], tf.stack(labels), \
                                    tf.stack(masks)[
                                        ..., tf.newaxis]

        return inputs_raw, labels, masks

    @tf.function
    def _validate_step(self, inputs: tf.Tensor, labels: tf.Tensor, masks: tf.Tensor,
                       loss_tensor: tf.keras.metrics.Mean,
                       training: bool = False) -> float:

        samples = self.call(inputs, training=training)
        loss = self.loss_fn(samples, labels, masks)
        loss_tensor.update_state(loss)
        return loss

    @tf.function
    def _train_step(self, inputs: tf.Tensor, labels: tf.Tensor, masks: tf.Tensor,
                    loss_tensor=tf.keras.metrics.Mean) -> float:
        with tf.GradientTape() as tape:
            loss = self._validate_step(inputs, labels, masks, loss_tensor, training=True)
        gradients = tape.gradient(loss, self.unet.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.unet.trainable_variables))
        return loss

    def loss_pn2v(self, samples, labels, masks):
        """
        The loss function as described in Eq. 7 of the paper.
        """

        likelihoods = self.noise_model.likelihood(labels, samples)
        likelihoods_avg = tf.math.log(
            tf.reduce_mean(likelihoods, axis=0, keepdims=True)[0, ...])

        # Average over pixels and batch
        masks = tf.squeeze(masks)
        loss = -tf.reduce_sum(likelihoods_avg * masks) / tf.reduce_sum(masks)
        return loss

    def loss_n2v(self, samples, labels, masks):
        """
        The loss function as described in Eq. 7 of the paper.
        """

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
        logging.info("Starting training loop")
        epochs = training_config.epochs
        train_writer = tf.summary.create_file_writer(
            os.path.join(self.model_path, 'logs', 'train'))
        valid_writer = tf.summary.create_file_writer(
            os.path.join(self.model_path, 'logs', 'valid'))

        losses = tf.keras.metrics.Mean(name='train_pn2v_loss')
        training_size = tf.data.experimental.cardinality(training_data).numpy()
        if validation_data is not None:
            val_losses = tf.keras.metrics.Mean(name='val_pn2v_loss')
            validation_size = tf.data.experimental.cardinality(validation_data)

        sample = next(iter(training_data))
        sample_images = sample['image']
        with train_writer.as_default():
            tf.summary.image('input', normalize_for_viz(sample_images), 0)
        best_loss = np.inf
        for epoch in range(epochs):
            desc = 'Epoch {} Training'
            train_bar = tqdm.tqdm(training_data,
                                  desc=desc.format(epoch), ncols=100, total=training_size)
            # for i in train_bar:
            for sample in train_bar:
                images = sample['image']
                labels = sample['label']
                masks = sample['mask']
                self._train_step(images, labels, masks, losses)
                train_bar.set_postfix(training_loss=str(losses.result().numpy())[:7],
                                      refresh=True)
                train_bar.set_description(desc.format(epoch))

            # Perform saving if loss is better
            current_loss = losses.result().numpy()
            if current_loss < best_loss:
                best_loss = current_loss
                tf.saved_model.save(self.unet, self.model_path)

            with train_writer.as_default():
                tf.summary.scalar('loss', losses.result().numpy(), epoch)
                if epoch % 5 == 0:
                    train_predictions = self.call(sample_images, training=False)
                    tf.summary.image('prediction',
                                     normalize_for_viz(
                                         train_predictions[..., tf.newaxis]),
                                     epoch)

            losses.reset_states()

            if validation_data is not None:
                desc = 'Epoch {} Validation'
                valid_bar = tqdm.tqdm(validation_data, desc=desc.format(epoch), ncols=100,
                                      total=validation_size)
                for sample in valid_bar:
                    val_images = sample['image']
                    val_labels = sample['label']
                    val_masks = sample['mask']
                    self._validate_step(val_images, val_labels, val_masks, val_losses)
                    valid_bar.set_postfix(
                        validation_loss=str(val_losses.result().numpy())[:7],
                        refresh=True)
                    valid_bar.set_description(desc.format(epoch))

                with valid_writer.as_default():
                    tf.summary.scalar('loss', losses.result().numpy(), epoch)
                    if epoch % 5 == 0:
                        val_predictions = self.call(sample_images, training=False)
                        tf.summary.image('prediction',
                                         normalize_for_viz(
                                             val_predictions[..., tf.newaxis]),
                                         epoch)

    def call(self, inputs, training=False):
        model_inputs = tf.convert_to_tensor((inputs - self.mean) / self.std)
        # Forward step
        # We found that this factor can speed up training
        outputs = self.unet(model_inputs, training=training) * 10.0

        samples = tf.transpose(outputs, (3, 0, 1, 2))
        # Denormalize
        samples = samples * self.std + self.mean
        samples = tf.reduce_mean(samples, axis=0)

        # TODO: check if this indexing fixes the training
        likelihoods = self.noise_model.likelihood(samples, inputs)
        mse_est = likelihoods * samples
        mse_est /= tf.reduce_sum(likelihoods)
        return mse_est
