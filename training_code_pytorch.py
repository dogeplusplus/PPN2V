import matplotlib.pyplot as plt
import numpy as np
from models.unet_pytorch import UNet

from pn2v import histNoiseModel
from pn2v import training

# See if we can use a GPU

path = "data/Confocal_MICE/raw/"

# Load the training image
data = np.load(path + 'training_raw.npy').astype(np.float32)

# We are loading the histogram from the 'Convallaria-1-CreateNoiseModel' notebook
histogram = np.load(path + 'noiseModel.npy')

# Create a NoiseModel object from the histogram.
noiseModel = histNoiseModel.NoiseModel(histogram, device='cpu')

numberOfRuns = 1  # If you want to run multiple repetitions of the training you can increse this.
for i in range(numberOfRuns):
    # Create a network with 800 output channels that are interpreted as samples from the prior.
    net = UNet(10, depth=2)

    path = "data/Confocal_MICE/raw/"

    # Split training and validation image.
    my_train_data = data[:-5].copy()
    #    np.random.shuffle(sample_data)
    my_val_data = data[-5:].copy()
    #    np.random.shuffle(my_val_data)

    # Start training.
    trainHist, valHist = training.trainNetwork(net=net, trainData=my_train_data, valData=my_val_data,
                                                postfix='mouse' + str(i), directory=path, noiseModel=noiseModel,
                                                numOfEpochs=200, stepsPerEpoch=50, patchSize=16,
                                                virtualBatchSize=20, batchSize=4, device='cpu')

    # Let's look at the training and validation loss
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(valHist, label='validation loss')
    plt.plot(trainHist, label='training loss')
    plt.legend()
    plt.show()
