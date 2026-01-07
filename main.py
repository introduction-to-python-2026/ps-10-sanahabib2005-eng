import numpy as np
from image_utils import load_image, edge_detection
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image

def main():
    # 1. Load original image
    img = load_image("IMG_1035.jpeg")

    # 2. Suppress noise with median filter
    clean_image = median(img, ball(3))

    # 3. Edge detection
    edges = edge_detection(clean_image)

    # 4. Convert to binary image
    threshold = 50
    edge_binary = edges > threshold

    # 5. Save edge-detected image
    edge_binary_uint8 = (edge_binary * 255).astype(np.uint8)
    edge_image = Image.fromarray(edge_binary_uint8)
    edge_image.save("my_edges.png")

if __name__ == "__main__":
    main()

