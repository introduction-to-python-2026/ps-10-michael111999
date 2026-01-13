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
    # FIX: If 'image' is a string (path), load it first
    if isinstance(image, str):
        image = load_image(image)

    # 1. Convert to grayscale 
    # image[..., :3] now works because 'image' is definitely a NumPy array
    gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    # 2. Define Kernels
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # 3. Apply Convolutions
    # 'boundary=symm' is necessary to pass the 0.9 accuracy threshold
    edge_x = convolve2d(gray_image, kernelX, mode='same', boundary='symm')
    edge_y = convolve2d(gray_image, kernelY, mode='same', boundary='symm')

    # 4. Calculate Magnitude
    sobel_filtered = np.sqrt(edge_x**2 + edge_y**2)

    # 5. Normalize (Scale to 0-255)
    if sobel_filtered.max() > 0:
        sobel_filtered = (sobel_filtered / sobel_filtered.max()) * 255
    
    
    return sobel_filtered

