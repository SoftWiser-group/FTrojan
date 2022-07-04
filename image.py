import math
from skimage import transform, data
import numpy as np
import cv2
import tensorflow as tf
from skimage.measure import compare_ssim
from tensorflow.keras.applications import InceptionV3
# from skimage.measure import compare_ssim
import bm3d
import scipy.signal

def RGB2YUV(x_rgb):
    x_yuv = np.zeros(x_rgb.shape, dtype=np.float)
    for i in range(x_rgb.shape[0]):
        img = cv2.cvtColor(x_rgb[i].astype(np.uint8), cv2.COLOR_RGB2YCrCb)
        x_yuv[i] = img
    return x_yuv

def YUV2RGB(x_yuv):
    x_rgb = np.zeros(x_yuv.shape, dtype=np.float)
    for i in range(x_yuv.shape[0]):
        img = cv2.cvtColor(x_yuv[i].astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        x_rgb[i] = img
    return x_rgb


def DCT(x_train, window_size):
    # x_train: (idx, w, h, ch)
    x_dct = np.zeros((x_train.shape[0], x_train.shape[3], x_train.shape[1], x_train.shape[2]), dtype=np.float)
    x_train = np.transpose(x_train, (0, 3, 1, 2))

    for i in range(x_train.shape[0]):
        for ch in range(x_train.shape[1]):
            for w in range(0, x_train.shape[2], window_size):
                for h in range(0, x_train.shape[3], window_size):
                    sub_dct = cv2.dct(x_train[i][ch][w:w+window_size, h:h+window_size].astype(np.float))
                    x_dct[i][ch][w:w+window_size, h:h+window_size] = sub_dct
    return x_dct            # x_dct: (idx, ch, w, h)


def IDCT(x_train, window_size):
    # x_train: (idx, ch, w, h)
    x_idct = np.zeros(x_train.shape, dtype=np.float)

    for i in range(x_train.shape[0]):
        for ch in range(0, x_train.shape[1]):
            for w in range(0, x_train.shape[2], window_size):
                for h in range(0, x_train.shape[3], window_size):
                    sub_idct = cv2.idct(x_train[i][ch][w:w+window_size, h:h+window_size].astype(np.float))
                    x_idct[i][ch][w:w+window_size, h:h+window_size] = sub_idct
    x_idct = np.transpose(x_idct, (0, 2, 3, 1))
    return x_idct


def Gaussian(x_train):
    # x_train: (idx, w, h, ch)
    x_train = x_train * 255
    for i in range(x_train.shape[0]):
        x_train[i] = cv2.GaussianBlur(x_train[i], (5, 5), sigmaX=0, sigmaY=0)
    x_train = x_train / 255.
    return x_train

def BM3D(x_train):
    x_train = x_train * 255
    for i in range(x_train.shape[0]):
        x_train[i] = bm3d.bm3d(x_train[i], sigma_psd=1)
    x_train = x_train / 255.
    return x_train

def Wiener(x_train):
    x_train = x_train * 255
    for i in range(x_train.shape[0]):
        img = np.transpose(x_train[i], (2, 0, 1))
        windows_size = (5, 5)
        img[0] = scipy.signal.wiener(img[0], windows_size)
        img[1] = scipy.signal.wiener(img[1], windows_size)
        img[2] = scipy.signal.wiener(img[2], windows_size)
        img = np.transpose(img, (1, 2, 0))
        x_train[i] = img
    x_train /= 255.
    return x_train

def PSNR(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def IS_score(img1, img2):
    img1 = transform.resize(img1, (299, 299))
    img1 = np.reshape(img1, (-1, 299, 299, 3))
    img2 = transform.resize(img2, (299, 299))
    img2 = np.reshape(img2, (-1, 299, 299, 3))
    model = InceptionV3(include_top=True, weights='imagenet',classes=1000)
    x1 = tf.keras.applications.inception_v3.preprocess_input(img1)
    x2 = tf.keras.applications.inception_v3.preprocess_input(img2)
    y1 = model(x1).numpy().reshape((-1))
    y2 = model(x2).numpy().reshape((-1))
    KL = 0.0
    for i in range(1000):
        KL += y1[i] * np.log(y1[i] / y2[i])
    return KL

def L2_Norm(img1, img2):
    return np.sqrt(np.sum((img1 - img2)**2))


def SSIM(img1, img2):
    res = compare_ssim(img1, img2, win_size=9, multichannel=True)
    # res = 0
    return res

def get_visual_values(imgs1, imgs2):
    iss, psnr, ssim, l2 = 0.0, 0.0, 0.0, 0.0
    for i in range(imgs1.shape[0]):
        psnr += PSNR(imgs1[i], imgs2[i])
        ssim += SSIM(imgs1[i], imgs2[i])
        iss += IS_score(imgs1[i], imgs2[i])
        l2 += L2_Norm(imgs1[i], imgs2[i])

    return psnr/imgs1.shape[0], ssim/imgs1.shape[0], iss/imgs1.shape[0], l2/imgs1.shape[0]

