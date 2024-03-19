import os
import pandas as pd
from numpy.random import seed
import ast
from glob import glob
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Flatten, BatchNormalization, Dense, Activation
from skimage.feature import graycomatrix, graycoprops
seed(1)
import tensorflow
tensorflow.random.set_seed(2)
import tensorflow as tf
from keras.layers import BatchNormalization
from glob import glob
import cv2
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import warnings
warnings.filterwarnings('ignore',category= UserWarning)
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Flatten, BatchNormalization, Dense, Activation

tf.disable_v2_behavior()

def f(x):
    return ast.literal_eval(x.rstrip('\r\n')) #convrt the annotations strings into their original data types integr for the usage in nprogram


def getBounds(geometry):#gettg the calculated bounding box coordinates xmin, ymin, xmax, ymax.

    try:
        arr = np.array(geometry).T
        xmin = np.min(arr[0])
        ymin = np.min(arr[1])
        xmax = np.max(arr[0])
        ymax = np.max(arr[1])
        return (xmin, ymin, xmax, ymax)
    except:
        return np.nan


def getWidth(bounds): #gettg widh
    try:
        (xmin, ymin, xmax, ymax) = bounds
        return np.abs(xmax - xmin)
    except:
        return np.nan


def getHeight(bounds):#gettg height points from annottaions
    try:
        (xmin, ymin, xmax, ymax) = bounds
        return np.abs(ymax - ymin)
    except:
        return np.nan


def upconv_concat(bottom_a, bottom_b, n_filter, k_size, stride, padding='VALID'):
    up_conv = tf.layers.conv2d_transpose(bottom_a, filters=n_filter, kernel_size=[k_size, k_size],
                                         strides=stride, padding=padding)
    return tf.concat([up_conv, bottom_b], axis=-1)


def conv_layer(bottom, k_size, num_outputs, stride, padding='SAME'):
    input_channels = int(bottom.get_shape()[-1])
    weights = tf.Variable(tf.truncated_normal(shape=[k_size, k_size, input_channels, num_outputs], dtype=tf.float32,
                                              stddev=np.sqrt(1.0 / (k_size * k_size * input_channels))))
    biases = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[num_outputs]))
    conv = tf.nn.conv2d(bottom, weights, strides=[1, stride, stride, 1], padding=padding)
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
    relu = tf.nn.relu(bias)

    return relu


def calc_loss(num_outputs, encoder_output, one_dim_kernel, sigma_pixel, spatial_kernel, batch_size):#lox calucltion
    num_sum = tf.constant(0.0, dtype=tf.float32)
    for depth in range(num_outputs):
        softmax_layer = encoder_output[:, :, :,
                        depth:depth + 1]  # take each channel of the output image from encoder_model
        extracted_pixels = tf.nn.conv2d(softmax_layer, one_dim_kernel, strides=[1, 1, 1, 1], padding='SAME')

        intensity_sq_dif = tf.squared_difference(extracted_pixels, softmax_layer)
        intensity_values = tf.exp(tf.divide(tf.negative(intensity_sq_dif), sigma_pixel))

        weights = tf.multiply(intensity_values, spatial_kernel)
        # Reshape Input Softmax Layer for correct dimensions
        u_pixels = tf.reshape(softmax_layer, [batch_size, 224, 224])

        # Calculate entire numerator
        numerator_inner_sum = tf.reduce_sum(tf.multiply(weights, extracted_pixels), axis=3)
        numerator_outer_sum = tf.multiply(u_pixels, numerator_inner_sum)
        numerator = tf.reduce_sum(numerator_outer_sum)

        # Calculate denominator
        denominator_inner_sum = tf.reduce_sum(weights, axis=3)
        denominator_outer_sum = tf.multiply(u_pixels, denominator_inner_sum)
        denominator = tf.reduce_sum(denominator_outer_sum)

        processed_value = numerator / denominator
        num_sum += processed_value

    return num_outputs - num_sum


def WNet_train(X_train):
    X_train = X_train.astype('float32') / 255.  # Standardizing the data
    X_train = np.reshape(X_train, (len(X_train), 224, 224, 1))  # reshape it to (1140, 224, 224, 1) for feeding to model

    height = width = 224
    channels = 1
    keep_prob = tf.placeholder_with_default(1.0, shape=())
    num_outputs = 3  # background, cell.
    num_epochs = 2000
    batch_size = 0
    r = 5
    sigma_dist = 4
    sigma_pixel = tf.square(tf.constant(10.0))
    input_img = tf.placeholder(dtype=tf.float32, shape=[None, height, width, channels], name='input_tensor')
    variance_epsilon = 0.0001
    with tf.name_scope('Encoder'):
        conv_1_1 = tf.layers.conv2d(input_img, 64, 3, padding='same', activation='relu')
        mean, variance = tf.nn.moments(conv_1_1, [0, 1, 2])
        # conv_1_1_bn = Batch_Normalization(conv_1_1, mean, variance, None, None, variance_epsilon)
        conv_1_1_bn = BatchNormalization(epsilon=variance_epsilon)(conv_1_1)

        conv_1_1_drop = tf.layers.dropout(conv_1_1_bn, keep_prob)
        conv_1_2 = tf.layers.conv2d(conv_1_1_drop, 64, 3, padding='same', activation='relu')
        mean, variance = tf.nn.moments(conv_1_2, [0, 1, 2])
        conv_1_2_bn = tf.nn.batch_normalization(conv_1_2, mean, variance, None, None, variance_epsilon)
        conv_1_2_drop = tf.layers.dropout(conv_1_2_bn, keep_prob)

        pool_1 = tf.layers.max_pooling2d(conv_1_2_drop, pool_size=2, strides=2, padding="valid")

        conv_2_1 = tf.layers.separable_conv2d(pool_1, 128, 3, padding='same', activation='relu')
        mean, variance = tf.nn.moments(conv_2_1, [0, 1, 2])
        conv_2_1_bn = tf.nn.batch_normalization(conv_2_1, mean, variance, None, None, variance_epsilon)
        conv_2_1_drop = tf.layers.dropout(conv_2_1_bn, keep_prob)

        conv_2_2 = tf.layers.separable_conv2d(conv_2_1_drop, 128, 3, padding='same', activation='relu')
        mean, variance = tf.nn.moments(conv_2_2, [0, 1, 2])
        conv_2_2_bn = tf.nn.batch_normalization(conv_2_2, mean, variance, None, None, variance_epsilon)
        conv_2_2_drop = tf.layers.dropout(conv_2_2_bn, keep_prob)

        pool_2 = tf.layers.max_pooling2d(conv_2_2_drop, pool_size=2, strides=2, padding="valid")

        conv_3_1 = tf.layers.separable_conv2d(pool_2, 256, 3, padding='same', activation='relu')
        mean, variance = tf.nn.moments(conv_3_1, [0, 1, 2])
        conv_3_1_bn = tf.nn.batch_normalization(conv_3_1, mean, variance, None, None, variance_epsilon)
        conv_3_1_drop = tf.layers.dropout(conv_3_1_bn, keep_prob)

        conv_3_2 = tf.layers.separable_conv2d(conv_3_1_drop, 256, 3, padding='same', activation='relu')
        mean, variance = tf.nn.moments(conv_3_2, [0, 1, 2])
        conv_3_2_bn = tf.nn.batch_normalization(conv_3_2, mean, variance, None, None, variance_epsilon)
        conv_3_2_drop = tf.layers.dropout(conv_3_2_bn, keep_prob)

        pool_3 = tf.layers.max_pooling2d(conv_3_2_drop, pool_size=2, strides=2, padding="valid")

        conv_4_1 = tf.layers.separable_conv2d(pool_3, 512, 3, padding='same', activation='relu')
        mean, variance = tf.nn.moments(conv_4_1, [0, 1, 2])
        conv_4_1_bn = tf.nn.batch_normalization(conv_4_1, mean, variance, None, None, variance_epsilon)
        conv_4_1_drop = tf.layers.dropout(conv_4_1_bn, keep_prob)

        conv_4_2 = tf.layers.separable_conv2d(conv_4_1_drop, 512, 3, padding='same', activation='relu')
        mean, variance = tf.nn.moments(conv_4_2, [0, 1, 2])
        conv_4_2_bn = tf.nn.batch_normalization(conv_4_2, mean, variance, None, None, variance_epsilon)
        conv_4_2_drop = tf.layers.dropout(conv_4_2_bn, keep_prob)

        pool_4 = tf.layers.max_pooling2d(conv_4_2_drop, pool_size=2, strides=2, padding="valid")

        conv_5_1 = tf.layers.separable_conv2d(pool_4, 1024, 3, padding='same', activation='relu')
        mean, variance = tf.nn.moments(conv_5_1, [0, 1, 2])
        conv_5_1_bn = tf.nn.batch_normalization(conv_5_1, mean, variance, None, None, variance_epsilon)
        conv_5_1_drop = tf.layers.dropout(conv_5_1_bn, keep_prob)

        conv_5_2 = tf.layers.separable_conv2d(conv_5_1_drop, 1024, 3, padding='same', activation='relu')
        mean, variance = tf.nn.moments(conv_5_2, [0, 1, 2])
        conv_5_2_bn = tf.nn.batch_normalization(conv_5_2, mean, variance, None, None, variance_epsilon)
        conv_5_2_drop = tf.layers.dropout(conv_5_2_bn, keep_prob)

        upconv_1 = upconv_concat(conv_5_2_drop, conv_4_2_drop, n_filter=512, k_size=2, stride=2)

        conv_6_1 = tf.layers.separable_conv2d(upconv_1, 512, 3, padding='same', activation='relu')
        mean, variance = tf.nn.moments(conv_6_1, [0, 1, 2])
        conv_6_1_bn = tf.nn.batch_normalization(conv_6_1, mean, variance, None, None, variance_epsilon)
        conv_6_1_drop = tf.layers.dropout(conv_6_1_bn, keep_prob)

        conv_6_2 = tf.layers.separable_conv2d(conv_6_1_drop, 512, 3, padding='same', activation='relu')
        mean, variance = tf.nn.moments(conv_6_2, [0, 1, 2])
        conv_6_2_bn = tf.nn.batch_normalization(conv_6_2, mean, variance, None, None, variance_epsilon)
        conv_6_2_drop = tf.layers.dropout(conv_6_2_bn, keep_prob)

        upconv_2 = upconv_concat(conv_6_2_drop, conv_3_2_drop, n_filter=256, k_size=2, stride=2)

        conv_7_1 = tf.layers.separable_conv2d(upconv_2, 256, 3, padding='same', activation='relu')
        mean, variance = tf.nn.moments(conv_7_1, [0, 1, 2])
        conv_7_1_bn = tf.nn.batch_normalization(conv_7_1, mean, variance, None, None, variance_epsilon)
        conv_7_1_drop = tf.layers.dropout(conv_7_1_bn, keep_prob)

        conv_7_2 = tf.layers.separable_conv2d(conv_7_1_drop, 256, 3, padding='same', activation='relu')
        mean, variance = tf.nn.moments(conv_7_2, [0, 1, 2])
        conv_7_2_bn = tf.nn.batch_normalization(conv_7_2, mean, variance, None, None, variance_epsilon)
        conv_7_2_drop = tf.layers.dropout(conv_7_2_bn, keep_prob)

        upconv_3 = upconv_concat(conv_7_2_drop, conv_2_2_drop, n_filter=128, k_size=2, stride=2)

        conv_8_1 = tf.layers.separable_conv2d(upconv_3, 128, 3, padding='same', activation='relu')
        mean, variance = tf.nn.moments(conv_8_1, [0, 1, 2])
        conv_8_1_bn = tf.nn.batch_normalization(conv_8_1, mean, variance, None, None, variance_epsilon)
        conv_8_1_drop = tf.layers.dropout(conv_8_1_bn, keep_prob)

        conv_8_2 = tf.layers.separable_conv2d(conv_8_1_drop, 128, 3, padding='same', activation='relu')
        mean, variance = tf.nn.moments(conv_8_2, [0, 1, 2])
        conv_8_2_bn = tf.nn.batch_normalization(conv_8_2, mean, variance, None, None, variance_epsilon)
        conv_8_2_drop = tf.layers.dropout(conv_8_2_bn, keep_prob)

        upconv_4 = upconv_concat(conv_8_2_drop, conv_1_2_drop, n_filter=64, k_size=2, stride=2)

        conv_9_1 = tf.layers.separable_conv2d(upconv_4, 64, 3, padding='same', activation='relu')
        mean, variance = tf.nn.moments(conv_9_1, [0, 1, 2])
        conv_9_1_bn = tf.nn.batch_normalization(conv_9_1, mean, variance, None, None, variance_epsilon)
        conv_9_1_drop = tf.layers.dropout(conv_9_1_bn, keep_prob)

        conv_9_2 = tf.layers.separable_conv2d(conv_9_1_drop, 64, 3, padding='same', activation='relu')
        mean, variance = tf.nn.moments(conv_9_2, [0, 1, 2])
        conv_9_2_bn = tf.nn.batch_normalization(conv_9_2, mean, variance, None, None, variance_epsilon)
        conv_9_2_drop = tf.layers.dropout(conv_9_2_bn, keep_prob)

        conv = tf.layers.conv2d(conv_9_2_drop, num_outputs, kernel_size=1, strides=1, padding='same', activation='relu')
        print(conv.shape, "conv")
        encoder_output = tf.nn.softmax(conv, axis=3, name="output_tensor")

    init = tf.global_variables_initializer()

    s = 2 * r + 1
    spatial_kernel = np.zeros((s, s), dtype=np.float32)
    for y in range(s):
        for x in range(s):
            # calculate squared euclidean distance
            dist = (x - r) * (x - r) + (y - r) * (y - r)
            if dist < (r * r):
                spatial_kernel[y][x] = np.exp((-dist) / sigma_dist)

    spatial_kernel = tf.constant(spatial_kernel.reshape(-1), dtype=tf.float32)

    # create one dimensional kernel

    s = 2 * r + 1
    one_dim_kernel = np.zeros((s, s, (s * s)))
    for i in range(s * s):
        one_dim_kernel[int(i / s)][i % s][i] = 1.0
    one_dim_kernel = one_dim_kernel.reshape(s, s, 1, (s * s))
    one_dim_kernel = tf.constant(one_dim_kernel, dtype=tf.float32)

    loss = calc_loss(num_outputs, encoder_output, one_dim_kernel, sigma_pixel, spatial_kernel, batch_size)
    soft_cut_norm_loss = tf.reduce_mean(loss)
    with tf.name_scope("optimizer"):
        norm_cut_opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(soft_cut_norm_loss)  # optimizer

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            print("epoch:", epoch)
            count = 0
            batch_start_index = 0
            while (count != 1):
                X_train_batch = X_train[batch_start_index: batch_start_index + batch_size]
                _, train_loss = sess.run([norm_cut_opt, soft_cut_norm_loss],
                                         feed_dict={input_img: X_train_batch, keep_prob: 0.7})
                batch_start_index += batch_size
                count += 1
            print("Train loss after epoch ", str(epoch), "is", str(train_loss))
        #     saved_path = saver.save(sess, './my-model')
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(),
                                                                        ["Encoder/output_tensor"])
        with tf.gfile.GFile("wnetmodel.pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())

    with tf.Session() as sess:
        sess.run(init)
        #     saver.restore(sess, './my-model')
        output = sess.run(encoder_output, feed_dict={input_img: X_train})
    return output



def preprocessing_object_detection():
    Features = []
    label = []
    df = pd.read_csv("Datasets/annotations.csv",
                     converters={'geometry': f})
    # Create bounds, width and height
    df.loc[:, 'bounds'] = df.loc[:, 'geometry'].apply(getBounds)
    df.loc[:, 'width'] = df.loc[:, 'bounds'].apply(getWidth)
    df.loc[:, 'height'] = df.loc[:, 'bounds'].apply(getHeight)

    image_files = glob('Datasets/images/*.*')
    unique, counts = np.unique(df['class'], return_counts=True)
    cnt = 1
    for image in image_files:
        filename = os.path.basename(image)
        all_indexes = np.where(df['image_id'] == filename)[0]
        img = cv2.imread(image)
        ### Image Enhancement
        image = img
        for ind in all_indexes:
            print("Preprocessing : "+str(cnt))
            img=image.copy()
            label.append(df['class'][ind])
            coordinates = df['bounds'][ind]
            xB = coordinates[2]
            xA = coordinates[0]
            yB = coordinates[3]
            yA = coordinates[1]
            cv2.rectangle(img, (xA, yA), (xB, yB), (0, 0, 0), 0)
            img = img[yA:yB, xA:xB]
            filename = 'Detected Objects\\img__' + str(cnt) + '.jpg'
            cv2.imwrite(filename, img)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            filenames = 'whiteobjects\\img__' + str(cnt) + '.jpg'
            cv2.imwrite(filenames, resized)
            cnt += 1
            # Resize image to (224, 224)
            Features.append(resized)  # append to list
    feat=np.asarray(Features)  # convert to numpy array
    lab = np.asarray(label)
    np.save("Features/Images.npy",feat)
    np.save("Features/Labels.npy",lab)

# preprocessing_object_detection()


def preprocessing_feature_extraction():
    # images = glob("Sharpened Objects/*.*")

    images = np.load("Features/sharpened_images.npy")
    images = images[:, :, :, 0]

    all_outputs = []
    cntt=0
    batch_size = 50
    for i in range(0, len(images), batch_size):
        end_index = min(i + batch_size, len(images))
        output = WNet_train(images[i:end_index])
        for k in output:
            all_outputs.append(k)
            plt.imshow(k)

    all_outputss = np.asarray(all_outputs)
    np.save("Sharpedsegmentatione.npy", all_outputs)

# preprocessing_feature_extraction()

# //////////shape and Structural Feauture Extrcation

def ldp_process(photo):
    def assign_bit(picture, x, y, c1, c2, d):  # assign bit according to degree and neighbouring pixel
        # a and b are 1 if increasing and 0 if decreasing
        if d == 0:
            a = 0
            b = 0
            try:
                if picture[c1][c2 + 1] >= picture[c1][c2]:
                    a = 1
                if picture[x][y + 1] >= picture[x][y]:
                    b = 1
            except:
                pass
        if d == 45:
            a = 0
            b = 0
            try:
                if picture[c1 - 1][c2 + 1] >= picture[c1][c2]:
                    a = 1
                if picture[x - 1][y + 1] >= picture[x][y]:
                    b = 1
            except:
                pass
        if d == 90:
            a = 0
            b = 0
            try:
                if picture[c1 - 1][c2] >= picture[c1][c2]:
                    a = 1
                if picture[x - 1][y] >= picture[x][y]:
                    b = 1
            except:
                pass
        if d == 135:
            a = 0
            b = 0
            try:
                if picture[c1 - 1][c2 - 1] >= picture[c1][c2]:
                    a = 1
                if picture[x - 1][y - 1] >= picture[x][y]:
                    b = 1
            except:
                pass
        if a == b:  # if monotonically increasing or decreasing than 0
            return "0"
        else:  # if turning point
            return "1"

        return bit

    def local_der_val(picture, x, y):  # calculating local derivative pattern value of a pixel
        thirtytwo_bit_binary = []
        centre = picture[x][y]
        c1 = x
        c2 = y
        decimal_val = 0
        # starting from top left,assigning bit to pixels clockwise at 0 degree
        thirtytwo_bit_binary.append(assign_bit(picture, x - 1, y - 1, c1, c2, 0))
        thirtytwo_bit_binary.append(assign_bit(picture, x - 1, y, c1, c2, 0))
        thirtytwo_bit_binary.append(assign_bit(picture, x - 1, y + 1, c1, c2, 0))
        thirtytwo_bit_binary.append(assign_bit(picture, x, y + 1, c1, c2, 0))
        thirtytwo_bit_binary.append(assign_bit(picture, x + 1, y + 1, c1, c2, 0))
        thirtytwo_bit_binary.append(assign_bit(picture, x + 1, y, c1, c2, 0))
        thirtytwo_bit_binary.append(assign_bit(picture, x + 1, y - 1, c1, c2, 0))
        thirtytwo_bit_binary.append(assign_bit(picture, x, y - 1, c1, c2, 0))

        # starting from top left,assigning bit to pixels clockwise at 45 degree
        thirtytwo_bit_binary.append(assign_bit(picture, x - 1, y - 1, c1, c2, 45))
        thirtytwo_bit_binary.append(assign_bit(picture, x - 1, y, c1, c2, 45))
        thirtytwo_bit_binary.append(assign_bit(picture, x - 1, y + 1, c1, c2, 45))
        thirtytwo_bit_binary.append(assign_bit(picture, x, y + 1, c1, c2, 45))
        thirtytwo_bit_binary.append(assign_bit(picture, x + 1, y + 1, c1, c2, 45))
        thirtytwo_bit_binary.append(assign_bit(picture, x + 1, y, c1, c2, 45))
        thirtytwo_bit_binary.append(assign_bit(picture, x + 1, y - 1, c1, c2, 45))
        thirtytwo_bit_binary.append(assign_bit(picture, x, y - 1, c1, c2, 45))

        # starting from top left,assigning bit to pixels clockwise at 90 degree
        thirtytwo_bit_binary.append(assign_bit(picture, x - 1, y - 1, c1, c2, 90))
        thirtytwo_bit_binary.append(assign_bit(picture, x - 1, y, c1, c2, 90))
        thirtytwo_bit_binary.append(assign_bit(picture, x - 1, y + 1, c1, c2, 90))
        thirtytwo_bit_binary.append(assign_bit(picture, x, y + 1, c1, c2, 90))
        thirtytwo_bit_binary.append(assign_bit(picture, x + 1, y + 1, c1, c2, 90))
        thirtytwo_bit_binary.append(assign_bit(picture, x + 1, y, c1, c2, 90))
        thirtytwo_bit_binary.append(assign_bit(picture, x + 1, y - 1, c1, c2, 90))
        thirtytwo_bit_binary.append(assign_bit(picture, x, y - 1, c1, c2, 90))

        # starting from top left,assigning bit to pixels clockwise at 135 degree
        thirtytwo_bit_binary.append(assign_bit(picture, x - 1, y - 1, c1, c2, 135))
        thirtytwo_bit_binary.append(assign_bit(picture, x - 1, y, c1, c2, 135))
        thirtytwo_bit_binary.append(assign_bit(picture, x - 1, y + 1, c1, c2, 135))
        thirtytwo_bit_binary.append(assign_bit(picture, x, y + 1, c1, c2, 135))
        thirtytwo_bit_binary.append(assign_bit(picture, x + 1, y + 1, c1, c2, 135))
        thirtytwo_bit_binary.append(assign_bit(picture, x + 1, y, c1, c2, 135))
        thirtytwo_bit_binary.append(assign_bit(picture, x + 1, y - 1, c1, c2, 135))
        thirtytwo_bit_binary.append(assign_bit(picture, x, y - 1, c1, c2, 135))

        str1 = ""
        l = str1.join(thirtytwo_bit_binary)  # 32 bit binary number
        decimal_val = int(l, 2)  # 32 bit binary to decimal number
        return decimal_val

    # m, n, _ = photo.shape
    m, n, = photo.shape
    # gray_scale = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)  # converting image to grayscale
    ldp_photo = np.zeros((m, n))
    # converting image to ldp
    for i in range(0, m):
        for j in range(0, n):
            ldp_photo[i, j] = local_der_val(photo, i, j)

    return ldp_photo
#canny shape
def edge_detection(img):
    a = (img * 255).astype("uint8")
    edges = cv2.Canny(a, 100, 200)  # Add threshold2 value (e.g., 200)
    return edges
def shape_structure_feautures(img):
    canny = edge_detection(img)
    ldp = ldp_process(img)
    combined_image = ldp + canny

    return combined_image

from glob import glob

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# GLCM

def GLCM_features(img):
    distance = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['correlation', 'homogeneity', 'contrast', 'energy', 'dissimilarity']

    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (5,5), 0)

    texture_features = []
    for i in range(0, blur.shape[0], 4):
        for j in range(0, blur.shape[1], 4):
            block = blur[i:i+4, j:j+4]

            glcm_mat = graycomatrix(block, distances=distance, angles=angles, symmetric=True, normed=True)
            block_glcm = np.hstack([graycoprops(glcm_mat, props).ravel() for props in properties])
            texture_features.append(block_glcm)

    return np.concatenate(texture_features)


# //LTP
def get_upper_pixel(img, center, x, y, t):
    t1 = center - t
    t2 = center + t
    s = 0
    if t1 < img[x][y] < t2:
        s = 0
    elif img[x][y] > t2:
        s = 1
    elif img[x][y] < t1:
        s = -1

    if s == 0:
        return 0
    elif s == 1:
        return 1
    elif s == -1:
        return 0

def get_lower_pixel(img, center, x, y, t):
    t1 = center - t
    t2 = center + t
    s = 0
    if t1 < img[x][y] < t2:
        s = 0
    elif img[x][y] > t2:
        s = 1
    elif img[x][y] < t1:
        s = -1

    if s == 0:
        return 0
    elif s == 1:
        return 0
    elif s == -1:
        return 1

def ltp_calculated_pixel_upper(img, x, y, t):
    center = img[x][y]
    val_ar = [
        get_upper_pixel(img, center, x - 1, y + 1, t),  # top_right
        get_upper_pixel(img, center, x, y + 1, t),      # right
        get_upper_pixel(img, center, x + 1, y + 1, t),  # bottom_right
        get_upper_pixel(img, center, x + 1, y, t),      # bottom
        get_upper_pixel(img, center, x + 1, y - 1, t),  # bottom_left
        get_upper_pixel(img, center, x, y - 1, t),      # left
        get_upper_pixel(img, center, x - 1, y - 1, t),  # top_left
        get_upper_pixel(img, center, x - 1, y, t)       # top
    ]

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val

def ltp_calculated_pixel_lower(img, x, y, t):
    center = img[x][y]
    val_ar = [
        get_lower_pixel(img, center, x - 1, y + 1, t),  # top_right
        get_lower_pixel(img, center, x, y + 1, t),      # right
        get_lower_pixel(img, center, x + 1, y + 1, t),  # bottom_right
        get_lower_pixel(img, center, x + 1, y, t),      # bottom
        get_lower_pixel(img, center, x + 1, y - 1, t),  # bottom_left
        get_lower_pixel(img, center, x, y - 1, t),      # left
        get_lower_pixel(img, center, x - 1, y - 1, t),  # top_left
        get_lower_pixel(img, center, x - 1, y, t)       # top
    ]

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val
def Modified_Object_Flow_Based_Ternary_Pattern(img):
    threshold_value = 5



    threshold_value = 5
    # Calculate upper pattern
    upper_pattern = np.zeros_like(img)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            upper_pattern[i, j] = ltp_calculated_pixel_upper(img, i, j, threshold_value)

    # Calculate lower pattern
    lower_pattern = np.zeros_like(img)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            lower_pattern[i, j] = ltp_calculated_pixel_lower(img, i, j, threshold_value)

    # Merge upper and lower patterns
    merged_pattern = cv2.addWeighted(upper_pattern, 0.5, lower_pattern, 0.5, 0)

    return merged_pattern

#//Resnet101

def create_resnet_model(input_shape):
    base_model = ResNet101(include_top=False, weights='imagenet', input_shape=input_shape)

    x = base_model.output
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(1024, kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(100, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def feature_extraction():
    # getimg = np.load("segmentation.npy")
    X = glob("Detected Objects/*.*")
    ter_list = []
    glcm_list = []
    for i in X:
        readd = cv2.imread(i)
        a = (i * 255).astype("uint8")
        gray_image = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        ter = shape_structure_feautures(gray_image)
        glcm = GLCM_features(gray_image)
        ter_list.append(ter)
        glcm_list.append(glcm)
    ter_array = np.array(ter_list)
    glcm_array = np.array(glcm_list)


    return ter_array,glcm_array

def ltp_execution():
    X = glob("Detected Objects/*.*")
    ltp_list = []
    for i in X:
        readd = cv2.imread(i)
        gray_image = cv2.cvtColor(readd, cv2.COLOR_BGR2GRAY)
        ltp = Modified_Object_Flow_Based_Ternary_Pattern(gray_image)
        ltp_list.append(ltp)
    ltp_array = np.array(ltp_list)
    return ltp_array

def predict_with_resnet():
    X = glob("Detected Objects/*.*")

    all_predictions = []
    for ig in X:
        img = cv2.imread(ig)
        input_shape = (img.shape[1], img.shape[2], img.shape[3])
        model = create_resnet_model(input_shape)
        predictions = model.predict(img)
        all_predictions.append(predictions)
    predictions_array = np.array(all_predictions)

    return predictions_array

def concatenate_arrays():
    ter_array, glcm_array = feature_extraction()
    ltp_array = ltp_execution()
    predictions_array = predict_with_resnet()

    # Concatenate the arrays along the last axis (axis=-1)
    joined_array = np.concatenate((ter_array, glcm_array, ltp_array, predictions_array), axis=-1)

    return joined_array






































