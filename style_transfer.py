#! /usr/bin/env python
import cv2
import numpy as np
from vgg19 import VGG_19
import keras.backend as K


def load_img(path):
    transpose_shape = (2, 0, 1)
    img_shape = (224, 224)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, img_shape).astype(np.float32)
    img = img.transpose(transpose_shape)
    img = np.expand_dims(img, axis=0)
    return img


class StyleTransfer:
    def __init__(self, style_img, content_img, noise_img):
        self._style_img = style_img
        self._content_img = content_img
        self._noise_img = noise_img
        self._net = VGG_19("vgg19_weights.h5")
        self._style_repr = None
        self._content_repr = None
        self.style_layers_weights = None
        self._alfa = 0.01
        self._beta = 0.99

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, val):
        self._beta = val
        self._alfa = 1 - val

    @property
    def alfa(self):
        return self._alfa

    @alfa.setter
    def alfa(self, val):
        self._alfa = val
        self._beta = 1 - val

    def eval_style_representation(self):
        self._style_layers = [3, 8, 17, 26, 25]
        self._style_repr = dict()
        for k in self._style_layers:
            get_activations = K.function([self._net.layers[0].input, K.learning_phase()], [self._net.layers[k].output])
            self._style_repr[k] = get_activations([self._style_img, 0])[0]

    def eval_content_representation(self):
        layer = 26
        get_activations = K.function([self._net.layers[0].input, K.learning_phase()], [self._net.layers[layer].output])
        self._content_repr = get_activations([self._content_img, 0])[0]

    def generate_new_image(self):
        #new error func:
        layers = range(27)
        F = dict()
        content_l = 26
        F = [self._net.layers[l].output for l in layers]
        L_content = self.content_mean_square_error(F[content_l], self._content_repr)
        weights = K.variable([0, 0.2, 0, 0.2, 0, 0, 0.2, 0, 0.2, 0,
                   0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2, 0, 0,
                   0.2, 0, 0.2, 0, 0.2, 0, 0.2, 0, 0, 0.2,
                   0, 0.2, 0, 0.2, 0, 0.2, 0, 0, 0, 0,
                   0, 0, 0])  # todo replace with style_layers_weights
        L_style = K.sum(weights[i] * self.gram_matrix_mean_squared_error(self._style_repr[i], F[i]) for i in self._style_layers)
        L_error = K.variable(self.alfa) * L_content + K.variable(self.beta) * L_style
        # content
        # layers = range(27)
        # F = dict()
        # content_l = 26
        # while True:
        #     # for l in layers:
        #     #     get_activations = K.function([self._net.layers[0].input, K.learning_phase()], [self._net.layers[l].output])
        #     #     F[l] = get_activations([noise_img, 0])
        #     L_content = self.content_mean_square_error(F[content_l][0], self._content_repr[0])
        #     #style
        #     weights = [0, 0.2, 0, 0.2, 0, 0, 0.2, 0, 0.2, 0, 0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2, 0, 0, 0.2, 0, 0.2, 0, 0.2, 0,
        #                0.2, 0, 0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2, 0, 0, 0, 0, 0, 0, 0] #todo replace with style_layers_weights
        #     L_style = 0
        #     for i in self._style_layers:
        #         L_style += weights[i] * self.gram_matrix_mean_squared_error(self._style_repr[i][0], F[i][0])
        #
        #     L_error = self.alfa * L_content + self.beta * L_style

    def gram_matrix_mean_squared_error(self, y_true, y_pred):
        y_pred = K.variable(value=y_pred)
        y_true = K.variable(value=y_true)
        _, filters, x_pos, y_pos = self.get_shape(y_pred)
        denominator = K.variable(value=(2 * filters * (x_pos * y_pos)) ** 2)
        y_pred = K.reshape(y_pred, (filters, x_pos * y_pos))
        y_true = K.reshape(y_true, (filters, x_pos * y_pos))
        return K.square(self.gram_matrix(y_pred) - self.gram_matrix(y_true)) / denominator

    def gram_matrix(self, x):
        return K.dot(x, K.transpose(x))

    def content_mean_square_error(self, y_true, y_pred):
        y_pred = K.variable(value=y_pred)
        y_true = K.variable(value=y_true)
        _, filters, x_pos, y_pos = self.get_shape(y_pred)
        y_pred = K.reshape(y_pred, (filters, x_pos * y_pos))
        y_true = K.reshape(y_true, (filters, x_pos * y_pos))
        return K.sum(K.square(y_pred - y_true))

    def get_shape(self, tensor):
        return tuple(map(lambda x: x.value, tensor.get_shape().dims))

if __name__ == '__main__':
    style_img = load_img("gogh.jpg")
    content_img = load_img("building.jpg")
    noise_img = load_img("whitenoise.png")
    st = StyleTransfer(style_img, content_img, noise_img)
    st.eval_style_representation()
    st.eval_content_representation()
    st.style_layers_weights = {k: 1 for k in [3, 8, 17, 26, 25]}
    st.generate_new_image()

    import ipdb; ipdb.set_trace()
