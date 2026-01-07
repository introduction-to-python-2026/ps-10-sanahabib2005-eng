from PIL import Image
import numpy as np
from scipy.ndimage import convolve

def load_image(path="IMG_1035.jpeg"):
    img = Image.open(path)
    return np.array(img)

def edge_detection(img):
    gray = img.mean(axis=2)
    kernelY = np.array([[ 1,  0, -1],
                        [ 2,  0, -2],
                        [ 1,  0, -1]])
    kernelX = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    edgeY = convolve(gray, kernelY, mode='constant', cval=0.0)
    edgeX = convolve(gray, kernelX, mode='constant', cval=0.0)
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    return edgeMAG


