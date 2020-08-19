import numpy as np
import logging.config
from os.path import join
from models.ppn2v import PPN2V
from utils.toolset import yaml2namespace
from utils.dataset import load_data, load_dataset
from tensorflow.python.profiler import profiler_v2

path = "data/Confocal_MICE/raw/training_raw.npy"

# Load the training image
data = np.load(path).astype(np.float32)

# We are loading the histogram from the 'Convallaria-1-CreateNoiseModel' notebook
# histogram = np.load(path + 'noiseModel.npy')

# Create a NoiseModel object from the histogram.
# noiseModel = hist_noise_model.NoiseModel(histogram)


logging.config.fileConfig("configs/logging.conf")

# TODO: how to deal with the noise model being a part of the model config as opposed to something generated from the data
model_config = yaml2namespace(join('unittests', 'assets', 'ppn2v_model.yaml'))
training_config = yaml2namespace(join('configs', 'training_config.yaml'))
# data, mean, std = load_data(data, batch_size=training_config.batch_size,
#                             patch_size=training_config.patch_size,
#                             num_pix=100 * 100 // 32, supervised=False)


train_data, val_data, mean, std = load_dataset('data/test_records')
model = PPN2V(model_config, mean, std)

profiler_v2.warmup()
profiler_v2.start(logdir='model_instances/cheese')
model.train(train_data, training_config)
profiler_v2.stop()
