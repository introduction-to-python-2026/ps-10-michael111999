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
    # 1. FIX: If image is 3D (RGB) from the median/ball filter, 
    # convert to grayscale IMMEDIATELY before doing math.
    if image.ndim == 3:
        # Standard ITU-R 601-2 luma transform
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    
    # 2. Convert to float64 to prevent the 'Square Overflow' bug
    # (uint8 max is 255; 255^2 is 65025, which crashes uint8)
    img_float = image.astype(np.float64)

    # 3. Standard Sobel Kernels
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # 4. Convolution with 'symm' boundary is REQUIRED for > 0.9 accuracy
    # 'fill' or 'wrap' will create edge artifacts that fail the test.
    edge_x = convolve2d(img_float, kernelX, mode='same', boundary='symm')
    edge_y = convolve2d(img_float, kernelY, mode='same', boundary='symm')

    # 5. Magnitude calculation
    magnitude = np.sqrt(edge_x**2 + edge_y**2)

    # 6. Strict Normalization to 0-255
    if magnitude.max() > 0:
        magnitude = (magnitude / magnitude.max()) * 255
    
    # 7. Cast to uint8 so that (true == edge_binary) in the test 
    # compares integers to integers, not floats to booleans.
    return magnitude.astype(np.uint8)
