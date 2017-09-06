#!/usr/bin/env python
import cv2, numpy as np, vgg19, utils.conv_display as cd

# Change variables to modify output
LAYER_NUMBER = 1
DIRECTORY = "test/layer_1"
# Create model
from keras.optimizers import SGD

model = vgg19.VGG_19('vgg19_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

# Load image
im = cv2.resize(cv2.imread('gogh.jpg'), (224, 224)).astype(np.float32)
im[:, :, 0] -= 103.939
im[:, :, 1] -= 116.779
im[:, :, 2] -= 123.68
im = im.transpose((2, 0, 1))
im = np.expand_dims(im, axis=0)

# Extract images from specified layer to desired output folder
cd.extract_layer_to_image(model, im, LAYER_NUMBER, DIRECTORY)
