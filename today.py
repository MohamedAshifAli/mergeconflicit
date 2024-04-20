from glob import glob
import cv2
import keras
import nilearn.plotting as nlplt
import nibabel as nib
import pandas as pd
from keras import backend as K
# from  mealpy1.music_based.SSE import dB
from imblearn.over_sampling import SMOTE
from keras import Model
from keras.applications import ResNet101
from matplotlib import pyplot as plt, gridspec
import math
from scipy.stats import skew, kurtosis, entropy
from skimage.util import montage
from skimage.transform import rotate
# import nilearn as nl
from sklearn.linear_model import SGDClassifier
from keras.utils import plot_model
from sklearn.model_selection import train_test_split, StratifiedKFold
# from mealpy1.swarm_based.GWO import  BaseGWO as GWO
# from mealpy1.swarm_based.BES import BaseBES as BEO
# from mealpy1.Prop import BaseProp as Prop
from termcolor import colored
import tensorflow as tf
# import lightgbm as lgb
from sklearn.metrics import  accuracy_score
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, Activation, Reshape, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, Add, multiply, Lambda

#
# TRAIN_DATASET_PATH = 'Datasets/MICCAI_BraTS_2019_Data_Training/Datasets/HGG/'
# #VALIDATION_DATASET_PATH = '../input/d/debobratachakraborty/brats2019-dataset/MICCAI_BraTS_2019_Data_Training/'
#
# test_image_flair=nib.load(TRAIN_DATASET_PATH + 'BraTS19_2013_10_1/BraTS19_2013_10_1_flair.nii').get_fdata()
# test_image_t1=nib.load(TRAIN_DATASET_PATH + 'BraTS19_2013_10_1/BraTS19_2013_10_1_t1.nii').get_fdata()
#
# test_image_t1ce=nib.load(TRAIN_DATASET_PATH + 'BraTS19_2013_10_1/BraTS19_2013_10_1_t1ce.nii').get_fdata()
# test_image_t2=nib.load(TRAIN_DATASET_PATH + 'BraTS19_2013_10_1/BraTS19_2013_10_1_t2.nii').get_fdata()
# test_mask=nib.load(TRAIN_DATASET_PATH + 'BraTS19_2013_10_1/BraTS19_2013_10_1_seg.nii').get_fdata()

DATASET_PATH ="Datasets/MICCAI_BraTS_2019_Data_Training/Datasets/HGG/"


def TestImageOutput():

    test_image1_flair = nib.load(DATASET_PATH + 'BraTS19_2013_2_1/BraTS19_2013_2_1_flair.nii').get_fdata()
    test_image1_t1 = nib.load(DATASET_PATH + 'BraTS19_2013_2_1/BraTS19_2013_2_1_t1.nii').get_fdata()
    test_image1_t1ce = nib.load(DATASET_PATH + 'BraTS19_2013_2_1/BraTS19_2013_2_1_seg.nii').get_fdata()
    test_image1_t2 = nib.load(DATASET_PATH + 'BraTS19_2013_2_1/BraTS19_2013_2_1_t1ce.nii').get_fdata()
    test_mask1 = nib.load(DATASET_PATH + 'BraTS19_2013_2_1/BraTS19_2013_2_1_t2.nii').get_fdata()
    fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(20, 10))
    slice_w = 25
    ax1.imshow(test_image1_flair[:, :, test_image1_flair.shape[0] // 2 - slice_w], cmap='gray')
    ax1.axis("off")
    ax1.set_title('Image flair')
    ax2.imshow(test_image1_t1[:, :, test_image1_t1.shape[0] // 2 - slice_w], cmap='gray')
    ax2.axis("off")
    ax2.set_title('Image t1')
    ax3.imshow(test_image1_t1ce[:, :, test_image1_t1ce.shape[0] // 2 - slice_w], cmap='gray')
    ax3.axis("off")
    ax3.set_title('Image t1ce')
    ax4.imshow(test_image1_t2[:, :, test_image1_t2.shape[0] // 2 - slice_w], cmap='gray')
    ax4.axis("off")
    ax4.set_title('Image t2')
    ax5.imshow(test_mask1[:, :, test_mask1.shape[0] // 2 - slice_w])
    ax5.axis("off")
    ax5.set_title('Mask')
    plt.savefig("ImageResults\\ActualImages.png", dpi=800)
    plt.show()
    plt.clf()
    # Skip 50:-50 slices since there is not much to see
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))
    ax1.imshow(rotate(montage(test_image1_t1[50:-50, :, :]), 90, resize=True), cmap='gray')
    plt.axis("off")
    plt.savefig("ImageResults\\FullSlices.png", dpi=800)
    plt.show()
    plt.clf()
    # Skip 50:-50 slices since there is not much to see
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))
    ax1.imshow(rotate(montage(test_mask1[60:-60, :, :]), 90, resize=True), cmap='gray')
    plt.axis("off")
    plt.savefig("ImageResults\\FullSlices_segmented.png", dpi=800)
    plt.show()

# TestImageOutput()




def preprocessg():
        ## Read the Brats_2019 Dataset
    dataset = glob("Datasets/MICCAI_BraTS_2019_Data_Training/Datasets/HGG"+"/**")

    features = []
    labels = []

    for i in range(10):
        ### Read all files inside the folder
        all_files = glob(dataset[i] + "/*.nii")

        ### from the list select flair and seg files
        flair_file = all_files[0]
        seg_file = all_files[1]

        ### Read nii file using the builtin function nibabel
        image = nib.load(flair_file).get_fdata()
        ### Read Ground Truth image
        gt=nib.load(seg_file).get_fdata()
        #### There may be 155 slices within we will take the ranges of 50 to 110
        for ii in range(50,110):
            #### image slices
            img = image[:, :, ii]
            ###  Mask slices
            msk = gt[:,:,ii]
            ### Get the label
            values = np.unique(msk)
            if values.any() >0:
                lab = max(values)
                if lab == 4:
                    lab = 3
                labels.append(lab)
                # Append segmentation mask instead of resized image
                features.append(msk)
                # # DEFINE seg-areas
                # SEGMENT_CLASSES = {
                #     0 : 'NOT tumor',
                #     1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
                #     2 : 'EDEMA',
                #     3 : 'ENHANCING' # original 4 -> converted into 3 later
                # }

            else:
                labels.append(0) # Non Tumor Labels
                features.append(msk)
    np.save("features.npy", features)
    np.save("labels.npy", labels)

    return features,labels
# preprocessg()


features = np.load("features.npy")
labels = np.load("labels.npy")
print()


# def Segmentation():
#     """
#         Perform segmentation on images using Hybrid WNet.
#
#         Returns:
#             all_outputs: Segmented images.
#         """
#     images = np.load("Features/Images.npy")[:] # Load images
#     images = images[:, :, :, 0]# Select the first channel
#     all_outputs = []
#     cntt=0
#     batch_size = 30 # Batch size for segmentation
#     for i in range(0, len(images), batch_size):
#         print("\033[93mSegmentation\033[0m : " + str(cntt))
#         end_index = min(i + batch_size, len(images)) # Print current batch index
#
#
#
#         output =Hybrid_WNet_train(images[i:end_index]) # Perform segmentation
#         cntt += 1
#         for k in output:
#             all_outputs.append(k)# Append segmented images to list
#             plt.imshow(k) # Save segmented images
#     all_outputss = np.asarray(all_outputs)
#     np.save("segmentatione.npy", all_outputss)
#     return all_outputs




def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def count_params(model):
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(p.shape) for p in model.weights])
    return param_count




answers = np.load('features.npy')

NOISE_DIM = 96


class MNIST(object):
    def __init__(self, batch_size, shuffle=False):
        """
        Construct an iterator object over the MNIST data

        Inputs:
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        train, _ = tf.keras.datasets.mnist.load_data()
        X, y = train
        X = X.astype(np.float32) / 255
        X = X.reshape((X.shape[0], -1))
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B))

mnist = MNIST(batch_size=25)
show_images(mnist.X[:25])


def sample_noise(batch_size, dim):
    """Generate random uniform noise from -1 to 1.

    Inputs:
    - batch_size: integer giving the batch size of noise to generate
    - dim: integer giving the dimension of the noise to generate

    Returns:
    TensorFlow Tensor containing uniform noise in [-1, 1] with shape [batch_size, dim]
    """
    # TODO: sample and return noise
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    noise = tf.random.uniform([batch_size, dim], minval=-1, maxval=1)
    return noise
    pass


def test_sample_noise():
    batch_size = 3
    dim = 4
    z = sample_noise(batch_size, dim)
    # Check z has the correct shape
    assert z.get_shape().as_list() == [batch_size, dim]
    # Make sure z is a Tensor and not a numpy array
    assert isinstance(z, tf.Tensor)
    # Check that we get different noise for different evaluations
    z1 = sample_noise(batch_size, dim)
    z2 = sample_noise(batch_size, dim)
    assert not np.array_equal(z1, z2)
    # Check that we get the correct range
    assert np.all(z1 >= -1.0) and np.all(z1 <= 1.0)
    print("All tests passed!")


test_sample_noise()


def discriminator():
    """Compute discriminator score for a batch of input images.

    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]

    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score
    for an image being real for each input image.
    """
    model = tf.keras.models.Sequential([
        # TODO: implement architecture
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        tf.keras.layers.InputLayer(784),
        tf.keras.layers.Dense(256),
        tf.keras.layers.LeakyReLU(0.01),
        tf.keras.layers.Dense(256),
        tf.keras.layers.LeakyReLU(0.01),
        tf.keras.layers.Dense(1)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ])
    return model


def test_discriminator(true_count=267009):
    model = discriminator()
    cur_count = count_params(model)
    if cur_count != true_count:
        print('Incorrect number of parameters in discriminator. {0} instead of {1}. Check your achitecture.'.format(
            cur_count, true_count))
    else:
        print('Correct number of parameters in discriminator.')


test_discriminator()


def generator(noise_dim=NOISE_DIM):
    """Generate images from a random noise vector.

    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]

    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    model = tf.keras.models.Sequential([
        # TODO: implement architecture
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        tf.keras.layers.InputLayer(noise_dim),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(784, activation=tf.nn.tanh)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ])
    return model


def test_generator(true_count=1858320):
    model = generator(4)
    cur_count = count_params(model)
    if cur_count != true_count:
        print(
            'Incorrect number of parameters in generator. {0} instead of {1}. Check your achitecture.'.format(cur_count,
                                                                                                              true_count))
    else:
        print('Correct number of parameters in generator.')


test_generator()


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: Tensor of shape (N, 1) giving scores for the real data.
    - logits_fake: Tensor of shape (N, 1) giving scores for the fake data.

    Returns:
    - loss: Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    real_loss = cross_entropy(tf.ones_like(logits_real), logits_real)
    fake_loss = cross_entropy(tf.zeros_like(logits_fake), logits_fake)
    loss = real_loss + fake_loss
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    loss = cross_entropy(tf.ones_like(logits_fake), logits_fake)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss

def test_discriminator_loss(logits_real, logits_fake, d_loss_true):
    d_loss = discriminator_loss(tf.constant(logits_real),
                                tf.constant(logits_fake))
    print("Maximum error in d_loss: %g"%rel_error(d_loss_true, d_loss))

test_discriminator_loss(answers['logits_real'], answers['logits_fake'],
                        answers['d_loss_true'])

def test_generator_loss(logits_fake, g_loss_true):
    g_loss = generator_loss(tf.constant(logits_fake))
    print("Maximum error in g_loss: %g"%rel_error(g_loss_true, g_loss))

test_generator_loss(answers['logits_fake'], answers['g_loss_true'])


# TODO: create an AdamOptimizer for D_solver and G_solver
def get_solvers(learning_rate=1e-3, beta1=0.5):
    """Create solvers for GAN training.

    Inputs:
    - learning_rate: learning rate to use for both solvers
    - beta1: beta1 parameter for both solvers (first moment decay)

    Returns:
    - D_solver: instance of tf.optimizers.Adam with correct learning_rate and beta1
    - G_solver: instance of tf.optimizers.Adam with correct learning_rate and beta1
    """
    D_solver = None
    G_solver = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    D_solver = tf.optimizers.Adam(learning_rate, beta1)
    G_solver = tf.optimizers.Adam(learning_rate, beta1)
    pass

    return D_solver, G_solver


# a giant helper function
def run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, \
              show_every=20, print_every=20, batch_size=128, num_epochs=10, noise_size=96):
    """Train a GAN for a certain number of epochs.

    Inputs:
    - D: Discriminator model
    - G: Generator model
    - D_solver: an Optimizer for Discriminator
    - G_solver: an Optimizer for Generator
    - generator_loss: Generator loss
    - discriminator_loss: Discriminator loss
    Returns:
        Nothing
    """
    mnist = MNIST(batch_size=batch_size, shuffle=True)

    iter_count = 0
    for epoch in range(num_epochs):
        for (x, _) in mnist:
            with tf.GradientTape() as tape:
                real_data = x
                logits_real = D(preprocess_img(real_data))

                g_fake_seed = sample_noise(batch_size, noise_size)
                fake_images = G(g_fake_seed)
                logits_fake = D(tf.reshape(fake_images, [batch_size, 784]))

                d_total_error = discriminator_loss(logits_real, logits_fake)
                d_gradients = tape.gradient(d_total_error, D.trainable_variables)
                D_solver.apply_gradients(zip(d_gradients, D.trainable_variables))

            with tf.GradientTape() as tape:
                g_fake_seed = sample_noise(batch_size, noise_size)
                fake_images = G(g_fake_seed)

                gen_logits_fake = D(tf.reshape(fake_images, [batch_size, 784]))
                g_error = generator_loss(gen_logits_fake)
                g_gradients = tape.gradient(g_error, G.trainable_variables)
                G_solver.apply_gradients(zip(g_gradients, G.trainable_variables))

            if (iter_count % show_every == 0):
                print('Epoch: {}, Iter: {}, D: {:.4}, G:{:.4}'.format(epoch, iter_count, d_total_error, g_error))
                imgs_numpy = fake_images.cpu().numpy()
                show_images(imgs_numpy[0:16])
                plt.show()
            iter_count += 1

    z = sample_noise(batch_size, noise_size)
    # generated images
    G_sample = G(z)
    print('Final images')
    show_images(G_sample[:16])
    plt.show()
# Make the discriminator
D = discriminator()

# Make the generator
G = generator()

# Use the function you wrote earlier to get optimizers for the Discriminator and the Generator
D_solver, G_solver = get_solvers()

# Run it!
run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss)
















































def Hybrid_Attention_Module_Segmentation(x):
    # Get Weights from previous layers
    all_x = x.node.layer.trainable_weights
    # From all weights select zeroth layer weights
    sel_x = all_x[0]
    w = np.array(sel_x)
    x = tf.convert_to_tensor(x)
    # feed selected weights into Triplet attention Module
    weights = HybridAttention(x).forward()
    # Reshape according to old weight
    weights = weights.reshape(w.shape[0], w.shape[1],w.shape[2],w.shape[3])
    # Convert it into tensor
    weights = tf.convert_to_tensor(weights)
    ##replace old weight by Triplet module generated weights
    all_x[0] = weights
    # updated all weights in x
    x.node.layer.trainable_weights_ = all_x
    return x

class HybridAttention:
    def __init__(self, x,use_skip_connection=False):
        self.x=x
        self.ca = ZeroChannelAttention()
        self.sa = ZeroSpatialAttention()
        self.use_skip_connection = use_skip_connection

    def forward(self):
        out = self.x
        out = out + out * self.sa.forward(out) if self.use_skip_connection else out * self.sa.forward(out)
        out =out[:self.x.shape[0], :self.x.shape[1], :self.x.shape[2], :self.x.shape[3]]
        ## Hybrid
        out=np.array(out)
        out=position_attention(out)
        return out
class ZeroChannelAttention:
    def __init__(self):
        self.avg_pool =  GlobalAveragePooling2D()
        self.max_pool = GlobalMaxPooling2D()

        self.sigmoid =Activation('sigmoid')

    def forward(self, x):
        self.avg_pool=self.avg_pool(x)
        self.max_pool=self.max_pool(x)
        self.x=Add()([self.avg_pool, self.max_pool])
        self.x=self.sigmoid(self.x)
        return self.x

class ZeroSpatialAttention:
    def __init__(self):
        self.sigmoid =Activation('sigmoid')

    def forward(self, X):
        self.avg_out =  Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(X)
        self.max_out= Lambda(lambda x: K.max(x, axis=3, keepdims=True))(X)
        self.x = Add()([self.avg_out, self.max_out])
        self.x = self.sigmoid(self.x)
        return self.x

class HybridAttention:
    def __init__(self, x,use_skip_connection=False):
        self.x=x
        self.ca = ZeroChannelAttention()
        self.sa = ZeroSpatialAttention()
        self.use_skip_connection = use_skip_connection

    def forward(self):
        out = self.x
        out = out + out * self.sa.forward(out) if self.use_skip_connection else out * self.sa.forward(out)
        out =out[:self.x.shape[0], :self.x.shape[1], :self.x.shape[2], :self.x.shape[3]]
        ## Hybrid
        out=np.array(out)
        out=position_attention(out)
        return out



def position_attention(input_feature, ratio=8):
    """
        Position attention mechanism.

        Args:
            input_feature: Input feature.
            ratio: Ratio for channel reduction.

        Returns:
            pam_feature: Output feature.
        """
    # channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel_axis = -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    pam_feature = Add()([avg_pool, max_pool])
    pam_feature = Activation('softmax')(pam_feature)

    return multiply([input_feature, pam_feature])

def Hybrid_Attention_Module(model):
    """
        Modify the weights of the model using hybrid attention mechanism.

        Args:
            model: Input model.

        Returns:
            model: Modified model.
        """
    #It will get weights from module
    weights = model.get_weights()
    weight = weights[0]
    tunedweight = Hybrid_Attention_Block(weight)
    weights[0]=tunedweight
    model.set_weights(weights)
    return model


def Hybrid_Attention_Block(w):
    """
        Hybrid attention block.
        Args:
            w: Input weights.
        Returns:
            newweight: Modified weights.
        """
    import tensorflow as tf
    x = tf.convert_to_tensor(w)
    feature = HybridAttention(x).forward()
    newweight = tf.reshape(feature, [w.shape[0], w.shape[1], w.shape[2],w.shape[3]])
    newweight = np.array(newweight)
    return newweight

def Hybrid_Attention_Module(model):
    """
        Modify the weights of the model using hybrid attention mechanism.

        Args:
            model: Input model.

        Returns:
            model: Modified model.
        """
    #It will get weights from module
    weights = model.get_weights()
    weight = weights[0]
    tunedweight = Hybrid_Attention_Block(weight)
    weights[0]=tunedweight
    model.set_weights(weights)
    return model

import tensorflow as tf
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'(((https://fromsmash.com/_X0zUuzCBz-dt
