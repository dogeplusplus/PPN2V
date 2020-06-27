#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from models.unet import Unet
from noise_models import hist_noise_model
from noise_models import training
from tifffile import imread
import glob

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--histogram", help="name of .npy-file containing the noise model histogram",
                    default='noise_model.npy')
parser.add_argument("--name", help="name of your network", default='N2V')
parser.add_argument("--dataPath", help="The path to your training image")
parser.add_argument("--fileName", help="name of your training image file", default="*.tif")
parser.add_argument("--validationFraction", help="Fraction of image you want to use for validation (percent)",
                    default=5.0, type=float)
parser.add_argument("--patchSizeXY", help="XY-patch_size of your training patches", default=64, type=int)
parser.add_argument("--epochs", help="number of training epochs", default=200, type=int)
parser.add_argument("--stepsPerEpoch", help="number training steps per epoch", default=50, type=int)
parser.add_argument("--batch_size", help="patch_size of your training batches", default=4, type=int)
parser.add_argument("--virtual_batch_size", help="patch_size of virtual batch", default=20, type=int)
parser.add_argument("--netDepth", help="depth of your U-Net", default=2, type=int)

parser.add_argument("--learning_rate", help="initial learning rate", default=1e-3, type=float)

parser.add_argument("--netKernelSize", help="Size of conv. kernels in first layer", default=3, type=int)
parser.add_argument("--n2vPercPix", help="percentage of pixels to manipulated by N2V", default=1.6, type=float)
parser.add_argument("--unet_n_first", help="number of feature channels in the first u-net layer", default=32, type=int)

if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()
print(args)


print("args", str(args.name))

####################################################
#           PREPARE TRAINING DATA
####################################################

path = args.dataPath
files = glob.glob(path + args.fileName)
# Load the training image
data = []
for f in files:
    data.append(imread(f).astype(np.float32))
    print('loading', f)

data = np.array(data)
print(data.shape)

if len(data.shape) == 4:
    data.shape = (data.shape[0] * data.shape[1], data.shape[2], data.shape[3])

print(data.shape)

####################################################
#           PREPARE Noise Model
####################################################

histogram = np.load(path + args.histogram)

# Create a NoiseModel object from the histogram.
noiseModel = hist_noise_model.NoiseModel(histogram)

####################################################
#           CREATE AND TRAIN NETWORK
####################################################

net = Unet(800, depth=args.netDepth)

# Split training and validation image.
my_train_data = data[:-5].copy()
np.random.shuffle(my_train_data)
my_val_data = data[-5:].copy()
np.random.shuffle(my_val_data)

# Start training.
train_hist, val_hist = training.train_network(net=net, train_data=my_train_data, val_data=my_val_data,
                                              postfix=args.name, directory=path, noise_model=noiseModel,
                                              epochs=args.epochs, steps_per_epoch=args.stepsPerEpoch,
                                              virtual_batch_size=args.virtualBatchSize, batch_size=args.batchSize,
                                              augment=False)
