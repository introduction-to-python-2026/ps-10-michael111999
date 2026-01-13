from image_utils import load_image, edge_detection
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
import numpy as np

import matplotlib.pyplot as plt
def load_image(path):
    # This function takes the string and returns the actual array
    img = Image.open(path)

    return np.array(img)


from scipy.signal import convolve2d
def edge_detection(image):
    # Ensure we are working with floats to prevent overflow
    if isinstance(image, str):
        image = load_image(image)
    
    img = image.astype(np.float64)

    # 1. Convert to grayscale FIRST if it's RGB
    if img.ndim == 3:
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

    # 2. Kernels
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # 3. Convolutions
    edge_x = convolve2d(img, kernelX, mode='same', boundary='symm')
    edge_y = convolve2d(img, kernelY, mode='same', boundary='symm')

    # 4. Magnitude & Normalization
    magnitude = np.sqrt(edge_x**2 + edge_y**2)
    
    # Scale to 0-255 and cast to uint8 for the test's "> 50" check
    if magnitude.max() > 0:
        magnitude = (magnitude / magnitude.max()) * 255
    
    return magnitude.astype(np.uint8)

