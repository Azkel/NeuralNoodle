#!/usr/bin/env python
import numpy as np
import cv2
import sys

img = cv2.imread(sys.argv[1])
output_img = cv2.pyrMeanShiftFiltering(img, 25, 100, 50)
cv2.imwrite(sys.argv[2], output_img)
