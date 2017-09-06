import utils.conv_display as cd
import cv2, numpy as np


def get_content_error_loss_for_layer(model, image1, image2, layer_number):
    layers1 = cd.get_activations(model, layer_number, image1)[0]
    layers2 = cd.get_activations(model, layer_number, image2)[0]

    # Create matrices
    filters_matrix_first_dimension = layers1.shape[1]
    filters_matrix_second_dimension = layers1.shape[2] * layers1.shape[3]
    array1 = np.zeros((filters_matrix_first_dimension, filters_matrix_second_dimension))
    array2 = np.zeros((filters_matrix_first_dimension, filters_matrix_second_dimension))
    for i in range(0, filters_matrix_first_dimension):
        array2[i] = layers2[0][i].flatten()
        array1[i] = layers1[0][i].flatten()
    # Count error Lcontent(~p, ~x, l)
    result_array = np.subtract(array1, array2)
    result_array = np.power(result_array, 2)
    error_sum = sum(sum(result_array))
    return error_sum / 2
