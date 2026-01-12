import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(path):
    # Open and ensure it is in RGB to standardize input
    img = Image.open(path).convert('RGB')
    return np.array(img)

def edge_detection(image):
    # 1. Convert to Grayscale (Luma)
    # This turns (H, W, 3) into (H, W)
    if image.ndim == 3:
        image = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    
    # 2. Define standard Sobel Kernels
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # 3. Apply Convolutions
    # Use 'same' to keep dimensions and 'symm' to match edge padding
    gx = convolve2d(image, kx, mode='same', boundary='symm')
    gy = convolve2d(image, ky, mode='same', boundary='symm')

    # 4. Calculate Magnitude
    magnitude = np.sqrt(gx**2 + gy**2)

    # 5. Normalize to 0-255
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude = (magnitude / mag_max) * 255
    
    # Ensure result is 2D (removes any accidental extra dimensions)
    return magnitude.squeeze()
