from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    # Use the 'path' variable passed into the function
    image = Image.open(path)
    return np.array(image)
def edge_detection(image):
    # 1. Convert to grayscale
    # If the image is already 2D (grayscale), skip this
    if len(image.shape) == 3:
        gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray_image = image

    # 2. Define Kernels
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # 3. Convolve
    # Use boundary='symm' to avoid dark edges that ruin the test score
    edges_x = convolve2d(gray_image, kernelX, mode='same', boundary='symm')
    edges_y = convolve2d(gray_image, kernelY, mode='same', boundary='symm')

    # 4. Calculate Magnitude
    magnitude = np.sqrt(edges_x**2 + edges_y**2)

    # 5. Normalize to 0-255 scale
    if magnitude.max() != 0:
        magnitude = (magnitude / magnitude.max()) * 255
    
    return magnitude
