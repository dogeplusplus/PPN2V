import os
import glob
import yaml
import datetime

import numpy as np
import tensorflow as tf

from typing import Tuple


def stratified_coords_2d(num_pix, shape):
    """
    Produce a list of approx. 'num_pix' random coordinate, sampled from 'shape' using stratified sampling.
    """
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
            if y < shape[0] and x < shape[1]:
                coords.append((y, x))
    return coords


def manipulate_pixels(image, num_pix):
    label = image
    max_a = image.shape[1] - 1
    max_b = image.shape[0] - 1
    mask = np.zeros(image.shape, dtype=np.uint8)
    hot_pixels = stratified_coords_2d(num_pix, image.shape)
    repl_grid = np.zeros(image.shape, dtype=np.uint8)
    for p in hot_pixels:
        a, b = p[1], p[0]

        roi_min_a = max(a - 2, 0)
        roi_max_a = min(a + 3, max_a)
        roi_min_b = max(b - 2, 0)
        roi_max_b = min(b + 3, max_b)
        roi = image[roi_min_b:roi_max_b, roi_min_a:roi_max_a]
        a_ = 2
        b_ = 2
        while a_ == 2 and b_ == 2:
            a_ = np.random.randint(0, roi.shape[1])
            b_ = np.random.randint(0, roi.shape[0])

        repl = roi[b_, a_]
        repl_grid[b, a] = repl
        mask[b, a] = 1
    return tf.where(mask > 0, repl_grid, image), label, mask


def manipulate_pixels_tf(image, num_pix):
    [image, label, mask, ] = tf.py_function(manipulate_pixels, [image, num_pix, ],
                                            [tf.float32, tf.float32, tf.float32])
    return image, label, mask


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

    return image, label, mask


def tf_augmentation(image, label, mask):
    [image, label, mask] = tf.py_function(augmentation, [image, label, mask],
                                          [tf.float32, tf.float32, tf.float32])
    return image, label, mask

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def features2example(image, label, mask):
    features = {
        'image': _bytes_feature(tf.image.encode_png(image, compression=1)),
        'label': _bytes_feature(tf.image.encode_png(label, compression=1)),
        'mask': _bytes_feature(tf.image.encode_png(mask, compression=1))
    }
    proto = tf.train.Example(features=tf.train.Features(feature=features))
    return proto.SerializeToString()


def _parse_function(feature):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'label': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'mask': tf.io.FixedLenFeature([], tf.string, default_value='')
    }
    parsed = tf.io.parse_single_example(feature, feature_description)
    features = {
        'image': tf.cast(tf.io.decode_png(parsed['image'], channels=1), tf.float32),
        'label': tf.io.decode_png(parsed['label'], channels=1),
        'mask': tf.io.decode_png(parsed['mask'], channels=1)
    }
    return features


def random_crop(image, size):
    height, width = image.shape
    y_coord = np.random.randint(size, height - size)
    x_coord = np.random.randint(size, width - size)
    return image[y_coord:y_coord + size, x_coord:x_coord + size]


def create_dataset(data_path, iterations, num_pix, destination, patch_size=120,
                   images_per_record=300,
                   validation_split=0.2):
    data = np.load(data_path).astype(np.uint8)
    mean = np.mean(data)
    std = np.std(data)
    val_split = int(data.shape[0] * validation_split)
    train_data = data[val_split:]
    val_data = data[:val_split]
    train_file = 'train_{}.tfrecord'
    val_file = 'valid_{}.tfrecord'
    config_file = 'data_config.yaml'

    os.makedirs(destination)
    for record_type, data_set in ((train_file, train_data), (val_file, val_data)):
        record_idx = 0
        record_counter = 0
        writer = tf.io.TFRecordWriter(
            os.path.join(destination, record_type.format(record_idx)))
        for _ in range(iterations):
            for sample in data_set:
                sample = random_crop(sample, patch_size)
                image, label, mask = manipulate_pixels(sample, num_pix)
                image, label, mask = image[..., np.newaxis], label[..., np.newaxis], mask[
                    ..., np.newaxis]
                example = features2example(image, label, mask)
                writer.write(example)
                record_counter += 1
                # Write to new TFRecord
                if record_counter > images_per_record:
                    record_counter = 0
                    record_idx += 1
                    writer = tf.io.TFRecordWriter(
                        os.path.join(destination, train_file.format(record_idx)))

    dataset_config = {
        'date_created': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'mean': float(mean),
        'std': float(std),
        'iterations': iterations,
        'num_pix': num_pix,
        'patch_size': patch_size,
        'images_per_record': images_per_record,
        'validation_split': validation_split
    }
    with open(os.path.join(destination, config_file), 'w') as f:
        yaml.dump(dataset_config, f)

def load_dataset(records_path, batch_size=4, shuffle_buffer=100):
    train_records = glob.glob(f'{records_path}/train_*.tfrecord')
    val_records = glob.glob(f'{records_path}/valid_*.tfrecord')
    dataset_config = f'{records_path}/data_config.yaml'
    train_data = tf.data.TFRecordDataset(train_records)
    val_data = tf.data.TFRecordDataset(val_records)
    datasets = {
        'train': train_data,
        'valid': val_data
    }

    for name in datasets.keys():
        datasets[name] = datasets[name].map(_parse_function)
        datasets[name] = datasets[name].map(tf_augmentation)
        datasets[name] = datasets[name].shuffle(buffer_size=shuffle_buffer)
        datasets[name] = datasets[name].prefetch(tf.data.experimental.AUTOTUNE)
        datasets[name] = datasets[name].batch(batch_size)

    with open(dataset_config, 'r') as f:
        config = yaml.full_load(f)

    return datasets['train'], datasets['valid'], config['mean'], config['std']


if __name__ == "__main__":
    path = "data/Confocal_MICE/raw/training_raw.npy"
    iterate = 10
    pixels = 100
    images_per_record = 1000
    destination = 'data/test_records'
    create_dataset(data_path=path, iterations=iterate, num_pix=pixels,
                   destination=destination, images_per_record=images_per_record)
