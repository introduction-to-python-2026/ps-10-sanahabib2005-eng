import numpy as np
from image_utils import load_image, edge_detection
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image

img = load_image("IMG_1035.jpeg")

clean_image = median(img, ball(3))

edges = edge_detection(clean_image)

threshold = 50
edge_binary = edges > threshold

edge_binary_uint8 = (edge_binary * 255).astype(np.uint8)
edge_image = Image.fromarray(edge_binary_uint8)
edge_image.save("my_edges.png")
