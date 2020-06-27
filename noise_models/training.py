import os
import tensorflow as tf
import numpy as np

from noise_models.utils import image2tensor


############################################
#   Training the network
############################################


def stratified_coords_2d(num_pix, shape):
    '''
    Produce a list of approx. 'numPix' random coordinate, sampled from 'shape' using stratified sampling.
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


def random_crop_fri(data, size, num_pix, supervised=False, counter=None, augment=True):
    '''
    Crop a patch from the next image in the dataset.
    The patches are augmented by randomly deciding to mirror them and/or rotating them by multiples of 90 degrees.
    
    Parameters
    ----------
    data: numpy array
        your dataset, should be a stack of 2D images, i.e. a 3D numpy array
    size: int
        width and height of the patch
    num_pix: int
        The number of pixels that is to be manipulated/masked N2V style.
    dataClean(optional): numpy array
        This dataset could hold your target image e.g. clean images.
        If it is not provided the function will use the image from 'image' N2V style
    counter (optional): int
        the index of the next image to be used. 
        If not set, a random image will be used.
    augment: bool
        should the patches be randomy flipped and rotated?
    
    Returns
    ----------
    img_out: numpy array
        Cropped patch from training image
    imgOutC: numpy array
        Cropped target patch. If dataClean was provided it is used as source.
        Otherwise its generated N2V style from the training set
    mask: numpy array
        An image holding marking which pixels should be used to calculate gradients (value 1) and which not (value 0)
    counter: int
        The updated counter parameter, it is increased by one.
        When the counter reaches the end of the dataset, it is reset to zero and the dataset is shuffled.
    '''

    if counter is None:
        index = np.random.randint(0, data.shape[0])
    else:
        if counter >= data.shape[0]:
            counter = 0
            np.random.shuffle(data)
        index = counter
        counter += 1

    if supervised:
        img = data[index, ..., 0]
        img_clean = data[index, ..., 1]
        manipulate = False
    else:
        img = data[index]
        img_clean = img
        manipulate = True

    img_out, img_out_c, mask = random_crop(img, size, num_pix,
                                           img_clean=img_clean,
                                           augment=augment,
                                           manipulate=manipulate)

    return img_out, img_out_c, mask, counter


def random_crop(img, size, num_pix, img_clean=None, augment=True, manipulate=True):
    '''
    Cuts out a random crop from an image.
    Manipulates pixels in the image (N2V style) and produces the corresponding mask of manipulated pixels.
    Patches are augmented by randomly deciding to mirror them and/or rotating them by multiples of 90 degrees.
    
    Parameters
    ----------
    img: numpy array
        your dataset, should be a 2D image
    size: int
        width and height of the patch
    num_pix: int
        The number of pixels that is to be manipulated/masked N2V style.
    img_clean (optional): numpy array
        This dataset could hold your target image e.g. clean images.
        If it is not provided the function will use the image from 'image' N2V style
    augment: bool
        should the patches be randomy flipped and rotated?
        
    Returns
    ----------    
    img_out: numpy array
        Cropped patch from training image with pixels manipulated N2V style.
    img_out_c: numpy array
        Cropped target patch. Pixels have not been manipulated.
    mask: numpy array
        An image marking which pixels have been manipulated (value 1) and which not (value 0).
        In N2V or PN2V only these pixels should be used to calculate gradients.
    '''

    assert img.shape[0] >= size
    assert img.shape[1] >= size

    x = np.random.randint(0, img.shape[1] - size)
    y = np.random.randint(0, img.shape[0] - size)

    img_out = img[y:y + size, x:x + size].copy()
    img_out_c = img_clean[y:y + size, x:x + size].copy()

    max_a = img_out.shape[1] - 1
    max_b = img_out.shape[0] - 1

    if manipulate:
        mask = np.zeros(img_out.shape)
        hot_pixels = stratified_coords_2d(num_pix, img_out.shape)
        for p in hot_pixels:
            a, b = p[1], p[0]

            roi_min_a = max(a - 2, 0)
            roi_max_a = min(a + 3, max_a)
            roi_min_b = max(b - 2, 0)
            roi_max_b = min(b + 3, max_b)
            roi = img_out[roi_min_b:roi_max_b, roi_min_a:roi_max_a]
            a_ = 2
            b_ = 2
            while a_ == 2 and b_ == 2:
                a_ = np.random.randint(0, roi.shape[1])
                b_ = np.random.randint(0, roi.shape[0])

            repl = roi[b_, a_]
            img_out[b, a] = repl
            mask[b, a] = 1.0
    else:
        mask = np.ones(img_out.shape)

    if augment:
        rot = np.random.randint(0, 4)
        img_out = np.array(np.rot90(img_out, rot))
        img_out_c = np.array(np.rot90(img_out_c, rot))
        mask = np.array(np.rot90(mask, rot))
        if np.random.choice((True, False)):
            img_out = np.array(np.flip(img_out))
            img_out_c = np.array(np.flip(img_out_c))
            mask = np.array(np.flip(mask))

    return img_out, img_out_c, mask


def training_pred(my_train_data, net, data_counter, size, bs, num_pix, noise_model, optimizer, augment=True,
                  supervised=True):
    '''
    This function will assemble a minibatch and process it using the a network.
    
    Parameters
    ----------
    my_train_data: numpy array
        Your training dataset, should be a stack of 2D images, i.e. a 3D numpy array
    net: a pytorch model
        the network we want to use
    data_counter: int
        The index of the next image to be used. 
    size: int
        Witdth and height of the training patches that are to be used.
    bs: int 
        The batch patch_size.
    num_pix: int
        The number of pixels that is to be manipulated/masked N2V style.
    augment: bool
        should the patches be randomly flipped and rotated?
    Returns
    ----------
    samples: pytorch tensor
        The output of the network
    labels: pytorch tensor
        This is the tensor that was is used a target.
        It holds the raw unmanipulated patches.
    masks: pytorch tensor
        A tensor marking which pixels have been manipulated (value 1) and which not (value 0).
        In N2V or PN2V only these pixels should be used to calculate gradients.
    dataCounter: int
        The updated counter parameter, it is increased by one.
        When the counter reaches the end of the dataset, it is reset to zero and the dataset is shuffled.
    '''

    # Init Variables
    inputs = []
    labels = []
    masks = []

    # Assemble mini batch
    for j in range(bs):
        im, l, m, data_counter = random_crop_fri(my_train_data,
                                                 size,
                                                 num_pix,
                                                 counter=data_counter,
                                                 augment=augment,
                                                 supervised=supervised)
        inputs.append(image2tensor(im))
        labels.append(image2tensor(l))
        masks.append(image2tensor(m))

    # Move to GPU
    inputs_raw, labels, masks = tf.stack(inputs)[..., tf.newaxis], tf.stack(labels), tf.stack(masks)[..., tf.newaxis]

    # Move normalization parameter to GPU
    std = tf.constant(net.std)
    mean = tf.constant(net.mean)

    model_inputs = tf.convert_to_tensor(inputs_raw - mean / std)
    # Forward step
    with tf.GradientTape(persistent=True) as tape:
        outputs = net(model_inputs) * 10.0  # We found that this factor can speed up training

        samples = tf.transpose(outputs, (3, 0, 1, 2))
        # Denormalize
        samples = samples * std + mean

        pn2v = (noise_model is not None) and (not supervised)

        likelihoods = noise_model.likelihood(labels, samples)
        likelihoods_avg = tf.math.log(tf.reduce_mean(likelihoods, axis=0, keepdims=True)[0, ...])

        # Average over pixels and batch
        masks = tf.squeeze(masks)
        loss = -tf.reduce_sum(likelihoods_avg * masks) / tf.reduce_sum(masks)

    gradients = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, net.trainable_variables))
    return loss


def loss_n2v(samples, labels, masks):
    '''
    The loss function as described in Eq. 7 of the paper.
    '''

    errors = (labels - tf.reduce_mean(samples, axis=0)) ** 2

    # Average over pixels and batch
    loss = tf.reduce_sum(errors * masks) / tf.reduce_sum(masks)
    return loss


def loss_pn2v(samples, labels, masks, noiseModel):
    '''
    The loss function as described in Eq. 7 of the paper.
    '''

    likelihoods = noiseModel.likelihood(labels, samples)
    likelihoods_avg = tf.math.log(tf.reduce_mean(likelihoods, axis=0, keepdims=True)[0, ...])

    # Average over pixels and batch
    masks = tf.squeeze(masks)
    loss = -tf.reduce_sum(likelihoods_avg * masks) / tf.reduce_sum(masks)
    return loss


def loss_obj(samples, labels, masks, noiseModel, pn2v, std=None):
    if pn2v:
        return loss_pn2v(samples, labels, masks, noiseModel)
    else:
        return loss_n2v(samples, labels, masks) / (std ** 2)


def train_network(net, train_data, val_data, noise_model, postfix,
                  directory='.',
                  epochs=200, steps_per_epoch=50,
                  batch_size=4, patch_size=100,
                  num_masked_pixels=100 * 100 / 32.0,
                  virtual_batch_size=20, val_size=20,
                  augment=True,
                  supervised=False
                  ):
    '''
    Train a network using PN2V

    Parameters
    ----------
    net:
        The network we want to train.
        The number of output channels determines the number of samples that are predicted.
    train_data: numpy array
        Our training image. A 3D array that is interpreted as a stack of 2D images.
    val_data: numpy array
        Our validation image. A 3D array that is interpreted as a stack of 2D images.
    noiseModel: NoiseModel
        The noise model we will use during training.
    postfix: string
        This identifier is attached to the names of the files that will be saved during training.
    directory: string
        The directory all files will be saved to.
    epochs: int
        Number of training epochs.
    steps_per_epoch: int
        Number of gradient steps per epoch.
    batch_size: int
        The batch patch_size, i.e. the number of patches processed simultainasly on the GPU.
    patch_size: int
        The width and height of the square training patches.
    num_masked_pixels: int
        The number of pixels that is to be manipulated/masked N2V style in every training patch.
    virtual_batch_size: int
        The number of batches that are processed before a gradient step is performed.
    val_size: int
        The number of validation patches processed after each epoch.
    augment: bool
        should the patches be randomly flipped and rotated?


    Returns
    ----------
    train_hist: numpy array
        A numpy array containing the avg. training loss of each epoch.
    val_hist: numpy array
        A numpy array containing the avg. validation loss after each epoch.
    '''

    # Calculate mean and std of image.
    combined = np.concatenate((train_data, val_data))
    net.mean = np.mean(combined)
    net.std = np.std(combined)

    # Everything that is processed by the net will be normalized and denormalized using these numbers.
    # TODO: Figure out how to get the reduce learning rate thing to work
    lr_schedule = 1e-4
    # lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='min', patience=10, factor=0.5, verbose=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    running_loss = 0.0
    data_counter = 0

    train_hist = []
    val_hist = []

    losses = []
    for epoch in range(epochs):
        for _ in range(steps_per_epoch):
            for a in range(virtual_batch_size):
                loss = training_pred(train_data,
                                     net,
                                     data_counter,
                                     patch_size,
                                     batch_size,
                                     num_masked_pixels,
                                     augment=augment,
                                     supervised=supervised,
                                     noise_model=noise_model,
                                     optimizer=optimizer)

                running_loss += loss
                losses.append(loss)

        running_loss = (np.mean(losses))
        losses = np.array(losses)
        print(f"Epoch: {epoch}, avg. loss: {np.mean(losses)} +-(2SEM) {2.0 * np.std(losses) / np.sqrt(losses.size)}")
        train_hist.append(np.mean(losses))
        tf.saved_model.save(net, os.path.join(directory, "last_" + postfix + ".net"))

        val_counter = 0
        net.trainable = False
        val_losses = []
        for i in range(val_size):
            loss = training_pred(val_data,
                                 net,
                                 val_counter,
                                 patch_size,
                                 batch_size,
                                 num_masked_pixels,
                                 augment=augment,
                                 supervised=supervised,
                                 noise_model=noise_model,
                                 optimizer=optimizer)
            val_losses.append(loss)
        net.trainable = True

        avg_val_loss = np.mean(val_losses)
        if len(val_hist) == 0 or avg_val_loss < np.min(np.array(val_hist)):
            tf.saved_model.save(net, os.path.join(directory, "best_" + postfix + ".net"))
        val_hist.append(avg_val_loss)
        np.save(os.path.join(directory, "history" + postfix + ".npy"),
                (np.array([np.arange(epoch), train_hist, val_hist])))

    return train_hist, val_hist
