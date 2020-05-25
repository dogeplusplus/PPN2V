import matplotlib.pyplot as plt
import numpy as np
from unet.model_tf import Unet

from pn2v_tf import hist_noise_model
from pn2v_tf import training

# See if we can use a GPU

path = "data/Confocal_MICE/raw/"

# Load the training data
data = np.load(path + 'training_raw.npy').astype(np.float32)

# We are loading the histogram from the 'Convallaria-1-CreateNoiseModel' notebook
histogram = np.load(path + 'noiseModel.npy')

# Create a NoiseModel object from the histogram.
noiseModel = hist_noise_model.NoiseModel(histogram)

numberOfRuns = 1  # If you want to run multiple repetitions of the training you can increse this.
for i in range(numberOfRuns):
    # Create a network with 800 output channels that are interpreted as samples from the prior.
    net = Unet(800, net_depth=3)

    path = "data/Confocal_MICE/raw/"

    # Split training and validation data.
    my_train_data = data[:-5].copy()
    # np.random.shuffle(my_train_data)
    my_val_data = data[-5:].copy()
    #    np.random.shuffle(my_val_data)

    # Start training.
    trainHist, valHist = training.train_network(net=net, train_data=my_train_data, val_data=my_val_data,
                                                postfix='mouse' + str(i), directory=path, noise_model=noiseModel,
                                                epochs=200, steps_per_epoch=50, patch_size=128,
                                                virtual_batch_size=20, batch_size=4)

    # Let's look at the training and validation loss
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(valHist, label='validation loss')
    plt.plot(trainHist, label='training loss')
    plt.legend()
    plt.show()
