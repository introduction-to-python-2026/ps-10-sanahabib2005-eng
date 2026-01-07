from PIL import Image
import numpy as np
from scipy.ndimage import convolve


def load_image(path):
    img = Image.open(path)      # קריאת התמונה
    img_array = np.array(img)   # המרה ל־NumPy array
    return img_array
from google.colab import files
uploaded = files.upload()
img = load_image("IMG_1035.jpeg")

print("Shape:", img.shape)
print("Dtype:", img.dtype)

import matplotlib.pyplot as plt
plt.imshow(img)
plt.axis("off")


def edge_detection(img):

    # 1. Convert to grayscale by averaging RGB channels
    gray = img.mean(axis=2)

    # 2. Define kernels
    kernelY = np.array([[ 1,  0, -1],
                        [ 2,  0, -2],
                        [ 1,  0, -1]])

    kernelX = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])

    # 3. Convolution with zero padding
    edgeY = convolve(gray, kernelY, mode='constant', cval=0.0)
    edgeX = convolve(gray, kernelX, mode='constant', cval=0.0)

    # 4. Edge magnitude
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG
edges = edge_detection(img)

print("Edge image shape:", edges.shape)
print("Edge image dtype:", edges.dtype)
import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.imshow(edges, cmap='gray')
plt.title("Edge Detection Result")
plt.axis("off")
edges = edge_detection(img)
assert edges.shape == img.shape[:2]
assert isinstance(edges, np.ndarray)

