import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from models.unet import Unet
from noise_models import hist_noise_model
from noise_models import training
from models.ppn2v import PPN2V
from utils.toolset import yaml2namespace
from utils.dataset import load_data
from utils.n2v_data_generator import N2V_DataGenerator

path = "data/Confocal_MICE/raw/"

# Load the training image
data = np.load(path + 'training_raw.npy').astype(np.float32)

# We are loading the histogram from the 'Convallaria-1-CreateNoiseModel' notebook
# histogram = np.load(path + 'noiseModel.npy')

# Create a NoiseModel object from the histogram.
# noiseModel = hist_noise_model.NoiseModel(histogram)

numberOfRuns = 1  # If you want to run multiple repetitions of the training you can increse this.
for i in range(numberOfRuns):
    # Create a network with 800 output channels that are interpreted as samples from the prior.
    # net = Unet(10, depth=2)

    # path = "image/Confocal_MICE/raw/"

    # Split training and validation image.
    # my_train_data = image[:-5].copy()
    # # np.random.shuffle(sample_data)
    # my_val_data = image[-5:].copy()
    # #    np.random.shuffle(my_val_data)
    #
    # # Start training.
    # trainHist, valHist = training.train_network(net=net, train_data=my_train_data, val_data=my_val_data,
    #                                             postfix='mouse' + str(i), directory=path, noise_model=noiseModel,
    #                                             epochs=200, steps_per_epoch=2, patch_size=64,
    #                                             virtual_batch_size=20, batch_size=4)
    #
    # # Let's look at the training and validation loss
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.plot(valHist, label='validation loss')
    # plt.plot(trainHist, label='training loss')
    # plt.legend()
    # plt.show()
    model_config = yaml2namespace(join('unittests', 'assets', 'ppn2v_model.yaml'))
    training_config = yaml2namespace(join('configs', 'training_config.yaml'))
    # TODO: Figure out what num pix should be
    data, mean, std = load_data(data, batch_size=training_config.batch_size, patch_size=training_config.patch_size,
                     num_pix=100 * 100 // 32, supervised=False)

    model = PPN2V(model_config, mean, std)
    model.train(data, training_config)
