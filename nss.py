import argparse

import cv2
import numpy as np
from keras.applications import vgg19
from scipy.optimize import fmin_l_bfgs_b
import keras.backend as K

# Weights for content & style
style_weight = 1
content_weight = 0.25
total_variation_weight = 1
image_width = 224
image_height = 224
iterations = 20


def deprocess_image(x):
    x = x.reshape((3, image_width, image_height))
    x = x.transpose((1, 2, 0))

    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def load_image_to_tensor(image_path):
    im = cv2.resize(cv2.imread(image_path), (224, 224)).astype(np.float32)
    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return vgg19.preprocess_input(im)


# Content loss is created using mean square error
def content_loss(content_image, result_image):
    return K.sum(K.square(result_image - content_image))


# Create gram matrix of input array
def gram_matrix(x):
    matrix = K.batch_flatten(x)
    return K.dot(matrix, K.transpose(matrix))


# Style loss is based on Gram-Matrices
def style_loss(style_image, result_image):
    style_gram_matrix = gram_matrix(style_image)
    result_gram_matrix = gram_matrix(result_image)
    return K.sum(K.square(style_gram_matrix - result_gram_matrix)) / (4. * 9 * ((image_width * image_height) ** 2))


# Creates a dictionary of layer outputs based on Content and Style images
def prepare_model_layers(style_path, content_path):
    style_image = K.variable(load_image_to_tensor(style_path))
    content_image = K.variable(load_image_to_tensor(content_path))
    result_image = K.placeholder((1, 3, image_width, image_height))
    tensor_mix = K.concatenate([content_image, style_image, result_image], axis=0)
    model = vgg19.VGG19(input_tensor=tensor_mix, weights='imagenet', include_top=False)
    return dict([(layer.name, layer.output) for layer in model.layers]), result_image


def total_variation_loss(x):
    a = K.square(x[:, :, :image_height - 1, :image_width - 1] - x[:, :, 1:, :image_width - 1])
    b = K.square(x[:, :, :image_height - 1, :image_width - 1] - x[:, :, :image_height - 1, 1:])
    return K.sum(K.pow(a + b, 1.25))


def loss_compute(layer_values, result_image):
    loss = K.variable(0.)
    layer_features = layer_values['block4_conv2']
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(base_image_features,combination_features)

    feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    for layer_name in feature_layers:
        layer_features = layer_values[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(feature_layers)) * sl
    loss += total_variation_weight * total_variation_loss(result_image)

    # get the gradients of the generated image wrt the loss
    grads = K.gradients(loss, result_image)

    outputs = [loss]
    if isinstance(grads, (list, tuple)):
        outputs += grads
    else:
        outputs.append(grads)

    f_outputs = K.function([result_image], outputs)
    return f_outputs


def eval_loss_and_grads(x, outputs_function):
    x = x.reshape((1, 3, image_width, image_height))
    outs = outputs_function([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


class Evaluator(object):

    def __init__(self, output_function):
        self.loss_value = None
        self.grads_values = None
        self.output_function = output_function

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x, self.output_function)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


def create_argparser():
    parser = argparse.ArgumentParser(description='Runs neural noodle')
    parser.add_argument('style_path', metavar='style_path', help='path to style image')
    parser.add_argument('content_path', metavar='content_path', help='path to content image')
    return parser


def main(style_path, content_path):
    output_image = "results/" + content_path[:content_path.rfind(".")].replace("/", "_") + "_" + style_path[:style_path.rfind(".")].replace("/","_") + "_iter{}.jpg"
    layers, result_image_tensor = prepare_model_layers(style_path, content_path)
    evaluator = Evaluator(loss_compute(layers, result_image_tensor))
    random_image = np.random.uniform(0, 255, (1, 3, image_width, image_height)) - 128.
    for i in range(iterations):
        print('Start of iteration', i)
        random_image, min_val, info = fmin_l_bfgs_b(evaluator.loss, random_image.flatten(),
                                                    fprime=evaluator.grads, maxfun=20)
        img = deprocess_image(random_image.copy())
        fname = output_image.format(i)
        cv2.imwrite(fname, img)
        print('Image saved as', fname)


if __name__ == '__main__':
    parser = create_argparser()
    args = parser.parse_args()
    main(args.style_path, args.content_path)
