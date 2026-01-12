import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(path):
    # The test needs to open whatever 'path' it provides, not a hardcoded name
    return np.array(Image.open(path))

def edge_detection(image):
    # 1. Convert to Grayscale
    # If the image is (H, W, 3), convert it. If it's already (H, W), leave it.
    if image.ndim == 3:
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    
    # 2. Define Kernels
    # These match the exact standard Sobel kernels
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # 3. Convolve
    # IMPORTANT: We use 'same' and 'symm' to match image size and border values
    # We cast to float to avoid overflow issues during math
    image = image.astype(float)
    gx = convolve2d(image, kernelX, mode='same', boundary='symm')
    gy = convolve2d(image, kernelY, mode='same', boundary='symm')

    # 4. Calculate Magnitude
    magnitude = np.sqrt(gx**2 + gy**2)

    # 5. Normalize
    # We scale to 0-255 so the "edge > 50" threshold in the test works correctly
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude = (magnitude / mag_max) * 255
    
    return magnitude
