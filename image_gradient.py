import cv2
import numpy as np
from vgg19 import VGG_19
from utils.conv_display import get_activations
import keras.backend as K


def transform_img(image):
    im = image.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return im


def get_features(image, net, layer):
    im = image.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)
    activations = get_activations(net, layer, im)
    return activations[0][0]


def my_mean_squared_error(y_true, y_pred):
    sh = y_pred.shape
    y_pred = y_pred.reshape((sh[0], sh[1], sh[2] * sh[3]))
    y_true = y_true.reshape((sh[0], sh[1], sh[2] * sh[3]))
    return K.mean(K.square(y_pred - y_true), axis=-1)


def L_content_derev(x, p, model, l):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[l].output])
    P = get_activations([transform_img(p), 0])
    F = get_activations([transform_img(x), 0])
    return my_mean_squared_error(F[0], P[0])


def deverivative(net, l, x):
    grad = K.function([net.layers[0].input, K.learning_phase()],
                      K.gradients([net.layers[l].output], [net.layers[0].input]))
    return grad([transform_img(x), 0])


def sgd_img(p, x, net):
    end_condition = False
    lambd = 0.02
    i = 0
    total = 60000
    layer = 26
    net.loss = my_mean_squared_error
    while not end_condition:
        err = deverivative(net, layer, x)[0]
        x = x - lambd * err.reshape(x.shape)
        if i % 10 == 0:
            cv2.imwrite("memes/meme_x2_{}.png".format(i), x)
        i = i + 1
        end_condition = i >= total
        print("{}/{}".format(i, total))


if __name__ == '__main__':
    # read images
    img = cv2.imread("gogh.jpg", cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224)).astype(np.float32)
    noise = cv2.imread("building.jpg", cv2.IMREAD_COLOR)
    noise = cv2.resize(noise, (224, 224)).astype(np.float32)

    # create net
    net = VGG_19("vgg19_weights.h5")
    new_img = sgd_img(img, noise, net)
