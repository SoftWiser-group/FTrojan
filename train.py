import numpy as np
from data import *
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from image import *
from model import *
from tensorflow import keras as keras
from tensorflow.keras.models import load_model as load_model
import matplotlib.pyplot as plt


def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 10:
        lrate = 0.0005
    elif epoch > 20:
        lrate = 0.0003
    return lrate

def train_clean():
    param = {
        "dataset": "CIFAR10",             # GTSRB, cifar10, MNIST, PubFig, ImageNet16
        "target_label": 8,              # target label
        "poisoning_rate": 0.02,         # rate of poisoned samples
        "label_dim": 10,
        "channel_list": [1,2],               # [0,1,2] means YUV channels, [1,2] means UV channels
        "magnitude": 20,
        "YUV": True,
        "clean_label": False,
        "window_size": 32,
        "pos_list": [(31, 31),(15, 15)],
    }

    x_train, y_train, x_test, y_test = get_data(param)

    param["input_shape"] = x_train.shape[1:]
    y_train = keras.utils.to_categorical(y_train, param["label_dim"])
    y_test = keras.utils.to_categorical(y_test, param["label_dim"])


    model = get_model(param)
    batch_size = 64

    filepath = "model/{}.hdf5".format(param["dataset"])
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='max')

    opt_rms = keras.optimizers.RMSprop(learning_rate=0.001, epsilon=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, batch_size=batch_size, steps_per_epoch=x_train.shape[0] // batch_size,
              epochs=60, verbose=1, validation_data=(x_test, y_test),
              callbacks=[keras.callbacks.LearningRateScheduler(lr_schedule), checkpoint])

    model.load_weights(filepath)
    scores_normal = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    print('\nTest on normal: %.3f loss: %.3f' % (scores_normal[1] * 100, scores_normal[0]))



def train():
    param = {
        "dataset": "CIFAR10",             # GTSRB, cifar10, MNIST, PubFig, ImageNet16
        "target_label": 8,              # target label
        "poisoning_rate": 0.02,         # ratio of poisoned samples
        "label_dim": 10,
        "channel_list": [1, 2],               # [0,1,2] means YUV channels, [1,2] means UV channels
        "magnitude": 20,
        "YUV": True,
        "clean_label": False,
        "window_size": 32,
        "pos_list": [(31, 31), (15, 15)],
    }

    x_train, y_train, x_test, y_test = get_data(param)
    x_train = poison(x_train, y_train, param)

    x_test_pos = impose(x_test.copy(), y_test.copy(), param)
    y_test_pos = np.array([[param["target_label"]]] * x_test_pos.shape[0], dtype=np.long)

    param["input_shape"] = x_train.shape[1:]
    y_train = keras.utils.to_categorical(y_train, param["label_dim"])
    y_test = keras.utils.to_categorical(y_test, param["label_dim"])
    y_test_pos = keras.utils.to_categorical(y_test_pos, param["label_dim"])

    model = get_model(param)
    batch_size = 32

    filepath = "model/{}.hdf5".format(digest(param))
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='max')


    if param["dataset"] in ["CIFAR10", "GTSRB"]:
        opt_rms = keras.optimizers.RMSprop(learning_rate=0.001, epsilon=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
        model.fit(x=x_train, y=y_train, batch_size=batch_size, steps_per_epoch=x_train.shape[0] // batch_size,
                  epochs=50, verbose=1, validation_data=(x_test, y_test),
                  callbacks=[keras.callbacks.LearningRateScheduler(lr_schedule), checkpoint])
    else:
        opt_rms = keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
        model.fit(x=x_train, y=y_train, batch_size=batch_size, steps_per_epoch=x_train.shape[0] // batch_size,
                  epochs=80, verbose=1, validation_data=(x_test, y_test),
                  callbacks=[keras.callbacks.LearningRateScheduler(lr_schedule), checkpoint])

    model.load_weights(filepath)
    scores_normal = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    scores_trojan = model.evaluate(x_test_pos, y_test_pos, batch_size=128, verbose=1)
    print('\nTest on normal: %.3f loss: %.3f' % (scores_normal[1] * 100, scores_normal[0]))
    print('\nTest on trojan: %.3f loss: %.3f' % (scores_trojan[1] * 100, scores_trojan[0]))




if __name__ == "__main__":
    # To avoid keras eat all GPU memory
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    train()





