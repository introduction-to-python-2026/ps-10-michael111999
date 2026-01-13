import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(path):
    """Loads an image and returns a numpy array."""
    # Ensure it's converted to RGB to handle PNGs or JPEGs consistently
    colorImg = np.array(Image.open(path).convert('RGB'))
    image = 0.3333 * colorImg[:,:,0] + 0.3333 * colorImg[:,:,1] + 0.3333 * colorImg[:,:,2]
    return image

def edge_detection(image):
    """Applies Sobel filter and returns magnitude."""
    # 1. Convert to grayscale
    if image.ndim == 3:
        image = 0.3333 * image[:,:,0] + 0.3333 * image[:,:,1] + 0.3333 * image[:,:,2]
    
    # 2. Define Sobel Kernels
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # 3. Convolve with symmetric boundary to avoid edge errors
    gx = convolve2d(image, kx, mode='same', boundary='symm')
    gy = convolve2d(image, ky, mode='same', boundary='symm')

    # 4. Calculate Magnitude
    magnitude = np.sqrt(gx**2 + gy**2)

    # # 5. Normalize to 0-255 range
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude = (magnitude / mag_max) * 255
    
    return magnitude
