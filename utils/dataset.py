import numpy as np
import tensorflow as tf

from typing import Tuple


def stratified_coords_2d(num_pix, shape):
    '''
    Produce a list of approx. 'num_pix' random coordinate, sampled from 'shape' using stratified sampling.
    '''
    box_size = np.round(np.sqrt(shape[0] * shape[1] / num_pix)).astype(np.int)
    coords = []
    box_count_y = int(np.ceil(shape[0] / box_size))
    box_count_x = int(np.ceil(shape[1] / box_size))
    for i in range(box_count_y):
        for j in range(box_count_x):
            y = np.random.randint(0, box_size)
            x = np.random.randint(0, box_size)
            y = int(i * box_size + y)
            x = int(j * box_size + x)
            if (y < shape[0] and x < shape[1]):
                coords.append((y, x))
    return coords


def manipulate_pixels(image, num_pix):
    label = image
    maxA = image.shape[1] - 1
    maxB = image.shape[0] - 1
    mask = np.zeros(image.shape)
    hot_pixels = stratified_coords_2d(num_pix, image.shape)
    repl_grid = np.zeros(image.shape, dtype=np.float32)
    for p in hot_pixels:
        a, b = p[1], p[0]

        roiMinA = max(a - 2, 0)
        roiMaxA = min(a + 3, maxA)
        roiMinB = max(b - 2, 0)
        roiMaxB = min(b + 3, maxB)
        roi = image[roiMinB:roiMaxB, roiMinA:roiMaxA]
        a_ = 2
        b_ = 2
        while a_ == 2 and b_ == 2:
            a_ = np.random.randint(0, roi.shape[1])
            b_ = np.random.randint(0, roi.shape[0])

        repl = roi[b_, a_]
        # TODO: setting array element with sequence error
        repl_grid[b, a] = repl
        mask[b, a] = 1.0
    return tf.where(mask > 0, repl_grid, image), label, mask


def augmentation(image, label, mask):
    prob = tf.random.uniform([])
    if prob > 0.5:
        image = np.flip(image, axis=0)
        label = np.flip(label, axis=0)
        mask = np.flip(mask, axis=0)

    rotation_factor = np.random.randint(0, 4)
    image = np.rot90(image, rotation_factor)
    label = np.rot90(label, rotation_factor)
    mask = np.rot90(mask, rotation_factor)

    return (image, label, mask)


def tf_augmentation(image, label, mask):
    [image, label, mask] = tf.py_function(augmentation, [image, label, mask], [tf.float32, tf.float32, tf.float32])
    return image, label, mask


def load_data(data_array: np.array, batch_size: int, patch_size: int, num_pix: int, supervised: bool = False,
              augment: bool = True) -> Tuple[tf.data.Dataset, float, float]:
    # TODO: determine if i need the dataset to have the channel on the end
    dataset = tf.data.Dataset.from_tensor_slices(data_array)

    # TODO: deal with this hacky nonense for the random crop
    # Crop first
    dataset = dataset.map(lambda x: tf.squeeze(tf.image.random_crop(x[..., tf.newaxis], (patch_size, patch_size, 1))))
    if supervised:
        images = dataset.map(lambda x: x[..., 0])
        labels = dataset.map(lambda x: x[..., 1])
        masks = dataset.map(lambda x: tf.ones_like(x))
        combined_dataset = tf.data.Dataset.zip((images, labels, masks))
    else:
        combined_dataset = dataset.map(lambda x: manipulate_pixels(x, num_pix))

    if augment:
        combined_dataset = combined_dataset.map(tf_augmentation)

    combined_dataset = combined_dataset.shuffle(buffer_size=100)
    combined_dataset = combined_dataset.batch(batch_size)
    combined_dataset = combined_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Return dataset with mean and std:
    mean = np.mean(data_array)
    std = np.std(data_array)

    return combined_dataset, mean, std
