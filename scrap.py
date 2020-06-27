import matplotlib.pyplot as plt
import numpy as np
import pickle

import noise_models.utils
import noise_models.hist_noise_model
from tifffile import imread

import numpy as np
import glob
from skimage import io
import matplotlib.pyplot as plt


# def loadFOV(number, path):
#     path = path + str(number) + "/*.png"
#     files = glob.glob(path)
#     image = []
#     print(path)
#     for f in files:
#         im_b = np.array(io.imread(f))
#         image.append(im_b)
#     image = np.array(image)
#     return image
#
#
# def loadFOVSingle(number, path):
#     path = path + str(number) + "/*.png"
#     files = glob.glob(path)
#     image = []
#     print(path)
#     f = files[0]
#     im_b = np.array(io.imread(f))
#     image.append(im_b)
#     image = np.array(image)
#     return image
#
#
# dataRaw = None
# dataGT = None
#
# dataTestRaw = None
# dataTestGT = None
#
# path = 'image/Confocal_MICE/raw/'
# for i in range(1, 20 + 1):
#     if i == 19:  # FOV 19 is reserved for testing
#         dataTestRaw = loadFOV(i, path).astype(np.float32)
#     else:
#         newa = loadFOV(i, path).astype(np.float32)
#         if dataRaw is None:
#             dataRaw = newa
#         else:
#             print(dataRaw.shape, newa.shape)
#             dataRaw = np.concatenate((dataRaw, newa), axis=0)
#     print(dataRaw.shape)
#
# dataRaw = np.array(dataRaw)
# np.save(path + '/training_raw.npy', dataRaw)
# np.save(path + '/test_raw.npy', dataTestRaw)
#
# path = 'image/Confocal_MICE/gt/'
# for i in range(1, 20 + 1):
#     if i == 19:  # FOV 19 is reserved for testing
#         dataTestGT = loadFOV(i, path).astype(np.float32)
#     else:
#         newa = loadFOV(i, path).astype(np.float32)
#         if dataGT is None:
#             dataGT = newa
#         else:
#             print(dataRaw.shape, newa.shape)
#             dataGT = np.concatenate((dataGT, newa), axis=0)
#         print(dataRaw.shape)
#
# dataGT = np.array(dataGT)
# dataRaw = np.array(dataRaw)
# np.save(path + '/training_gt.npy', dataGT)
# np.save(path + '/test_gt.npy', dataTestGT)

path="data/Confocal_MICE/raw/"

observation= np.load(path+'training_raw.npy')

# The image contains 50 images of a static sample.
# The authors provide the groundturth as the average of the 50 images
signal= np.load(path+'../gt/training_gt.npy')
print(observation.shape, signal.shape)

# Let's look the raw image and our pseudo ground truth signal
plt.figure(figsize=(12, 12))
plt.subplot(1, 2, 2)
plt.title(label='average (ground truth)')
plt.imshow(signal[0],cmap='gray')
plt.subplot(1, 2, 1)
plt.title(label='single raw image')
plt.imshow(observation[0],cmap='gray')
plt.show()

# We set the range of values we want to cover with our model.
# The pixel intensities in the images you want to denoise have to lie within this range.
# The dataset is clipped to values between 0 and 255.
minVal, maxVal =0, 256
bins = 256

# We are creating the histogram.
# This can take a minute.
histogram = noise_models.hist_noise_model.create_histogram(bins, minVal, maxVal, observation, signal)

# Saving histogram to disc.
np.save(path+'noiseModel.npy', histogram)

histogram=histogram[0]

plt.xlabel('observation bin')
plt.ylabel('signal bin')
plt.imshow(histogram**0.25, cmap='gray')
plt.show()

xvals=np.arange(bins)/float(bins)*(maxVal-minVal)+minVal
plt.xlabel('observation')
plt.ylabel('probability density')

# We will now look at the noise distributions for different signals s_i,
# by plotting individual rows of the histogram
# Note that the image is clipped at 255.

index=5
s=((index+0.5)/float(bins)*(maxVal-minVal)+minVal)
plt.plot(xvals,histogram[index,:], label='bin='+str(index)+' signal='+str(np.round(s,2)))

index=25
s=((index+0.5)/float(bins)*(maxVal-minVal)+minVal)
plt.plot(xvals,histogram[index,:], label='bin='+str(index)+' signal='+str(np.round(s,2)))

index=50
s=((index+0.5)/float(bins)*(maxVal-minVal)+minVal)
plt.plot(xvals,histogram[index,:], label='bin='+str(index)+' signal='+str(np.round(s,2)))

index=100
s=((index+0.5)/float(bins)*(maxVal-minVal)+minVal)
plt.plot(xvals,histogram[index,:], label='bin='+str(index)+' signal='+str(np.round(s,2)))

index=150
s=((index+0.5)/float(bins)*(maxVal-minVal)+minVal)
plt.plot(xvals,histogram[index,:], label='bin='+str(index)+' signal='+str(np.round(s,2)))

index=200
s=((index+0.5)/float(bins)*(maxVal-minVal)+minVal)
plt.plot(xvals,histogram[index,:], label='bin='+str(index)+' signal='+str(np.round(s,2)))

index=250
s=((index+0.5)/float(bins)*(maxVal-minVal)+minVal)
plt.plot(xvals,histogram[index,:], label='bin='+str(index)+' signal='+str(np.round(s,2)))

plt.legend()
plt.show()
