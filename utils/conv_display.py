# Based on: 
# https://github.com/fchollet/keras/issues/2890
# https://github.com/fchollet/keras/issues/41
from keras import backend as K
import cv2
import numpy as np


def extract_layer_to_image(model, input_data, layer_number, image_name):
    activations = get_activations(model, layer_number, input_data)
    for i in range(len(activations[0])):
        for j in range(len(activations[0][i])):
            cv2.imwrite(image_name + "_" + str(i) + "_" + str(j) + ".jpg", activations[0][i][j])


def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])
    activations = get_activations([X_batch, 0])
    return activations
