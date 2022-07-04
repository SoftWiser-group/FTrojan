from os import listdir
from model import get_model
import random
import idx2numpy
import numpy as np
import tensorflow.keras as keras
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from image import *
import random
import tensorflow as tf
from hashlib import md5
from PIL import Image

def get_data(param):
    prefix = "/home/wt/code/DCT_Attack/"
    if param["dataset"] == "MNIST":
        train_image = open(prefix + "data/{}/train-images-idx3-ubyte".format(param["dataset"]), "rb")
        x_train = idx2numpy.convert_from_file(train_image).reshape((-1, 28, 28, 1)).astype(np.float) / 255

        train_label = open(prefix + "data/{}/train-labels-idx1-ubyte".format(param["dataset"]), "rb")
        y_train = idx2numpy.convert_from_file(train_label).astype(np.long).reshape((-1, 1))

        test_image = open(prefix + "data/{}/t10k-images-idx3-ubyte".format(param["dataset"]), "rb")
        x_test = idx2numpy.convert_from_file(test_image).reshape((-1, 28, 28, 1)).astype(np.float) / 255

        test_label = open(prefix + "data/{}/t10k-labels-idx1-ubyte".format(param["dataset"]), "rb")
        y_test = idx2numpy.convert_from_file(test_label).astype(np.long).reshape((-1, 1))
        # x_train = padding_MNIST(x_train)
        # x_test = padding_MNIST(x_test)

    if param["dataset"] == "CIFAR10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype(np.float) / 255.
        x_test = x_test.astype(np.float) / 255.

    if param["dataset"] == "GTSRB":
        # npz has been normalized to (0,1)
        data = np.load(prefix + "data/GTSRB43.npz")
        x_train, y_train, x_test, y_test = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]

    if param["dataset"] == "ImageNet16":
        # npz has been normalized to (0,1)
        data = np.load(prefix + "data/ImageNet16.npz")
        x_train, y_train, x_test, y_test = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]

    if param["dataset"] == "Intel":
        # npz have been normalized to (0,1)
        data = np.load(prefix + "data/Intel.npz")
        x_train, y_train, x_test, y_test = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]

    if param["dataset"] == "PubFig":
        # npz have been normalized to (0,1)
        data = np.load(prefix + "data/PubFig16.npz")
        x_train, y_train, x_test, y_test = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]

    return x_train, y_train, x_test, y_test

# SIG
def poison_sin_one(img, delta=20, f=6, debug=False):  # hwc
    img = np.float32(img)
    pattern = np.zeros_like(img)
    m = pattern.shape[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)

    img = img + pattern
    img = np.where(img > 255, 255, img)
    img = np.where(img < 0, 0, img)
    return img

# Generate Refool and insert
def poison_refool(img):
    return None


def poison(x_train, y_train, param):
    target_label = param["target_label"]
    num_images = int(param["poisoning_rate"] * y_train.shape[0])

    if param["clean_label"]:
        index = np.where(y_train == target_label)
        index = index[0]
        index = index[:num_images]

        x_train[index] = poison_frequency(x_train[index], y_train[index], param)
        return x_train

    else:
        index = np.where(y_train != target_label)
        index = index[0]
        index = index[:num_images]
        x_train[index] = poison_frequency(x_train[index], y_train[index], param)
        y_train[index] = target_label
        return x_train


def poison_frequency(x_train, y_train, param):
    if x_train.shape[0] == 0:
        return x_train

    if param["clean_label"]:
        model = get_model(param)
        model.load_weights("model/{}.hdf5".format(param["dataset"]))
        loss_object = tf.keras.losses.CategoricalCrossentropy()

        for batch in range(0, x_train.shape[0], 100):
            with tf.GradientTape() as tape:
                images = tf.convert_to_tensor(x_train[batch:batch+100], dtype=tf.float32)
                tape.watch(images)
                prediction = model(images)
                y_target = keras.utils.to_categorical(y_train[batch:batch+100], param["label_dim"])
                y_target = tf.convert_to_tensor(y_target, dtype=tf.float32)
                loss = loss_object(y_target, prediction)
            gradient = tape.gradient(loss, images)
            gradient = np.array(gradient, dtype=np.float)

            for i in range(images.shape[0]):
                x_train[batch+i] = x_train[batch+i] + (param["degree"] / 255.) * gradient[i] / (1e-20 + np.sqrt(np.sum(gradient[i] * gradient[i])))

    x_train *= 255.
    if param["YUV"]:
        # transfer to YUV channel
        x_train = RGB2YUV(x_train)

    # transfer to frequency domain
    x_train = DCT(x_train, param["window_size"])  # (idx, ch, w, h)

    # plug trigger frequency
    for i in range(x_train.shape[0]):
        for ch in param["channel_list"]:
            for w in range(0, x_train.shape[2], param["window_size"]):
                for h in range(0, x_train.shape[3], param["window_size"]):
                    for pos in param["pos_list"]:
                        x_train[i][ch][w + pos[0]][h + pos[1]] += param["magnitude"]

    x_train = IDCT(x_train, param["window_size"])  # (idx, w, h, ch)

    if param["YUV"]:
        x_train = YUV2RGB(x_train)
    x_train /= 255.
    x_train = np.clip(x_train, 0, 1)
    return x_train


def impose(x_train, y_train, param):
    x_train = poison_frequency(x_train, y_train, param)
    return x_train


def digest(param):
    txt = ""
    txt += param["dataset"]
    txt += str(param["target_label"])
    txt += str(param["poisoning_rate"])
    txt += str(param["label_dim"])
    txt += "".join(str(param["channel_list"]))
    txt += str(param["window_size"])
    txt += str(param["magnitude"])
    txt += str(param["YUV"])
    txt += str(param["clean_label"])
    txt += "".join(str(param["pos_list"]))
    hash_md5 = md5()
    hash_md5.update(txt.encode("utf-8"))
    return hash_md5.hexdigest()
