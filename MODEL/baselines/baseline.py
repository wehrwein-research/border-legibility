import sys,os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.image as img
import cv2
import imutils

""" Pyramid gradient/gradient norm models

    Author: Thomas Nelson
"""

def norm_of_gradients(image):
    if type(image) == str:
        image = cv2.imread(image)
    lap = cv2.Laplacian(image,cv2.CV_64F,ksize=3) 
    lap = np.uint8(np.absolute(lap))
    norm = np.mean(lap)
    return norm

def pyramid_gradient(tile):
    image = img.imread(tile)
    score = 0
    for (i, resized) in enumerate(pyramid(image, scale=1.5)):
        norm = norm_of_gradients(resized)
        try:
            score += math.pow(norm, 2)
        except:
            print("gradient not found in image")
                   
    return math.sqrt(score)

def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield image
    