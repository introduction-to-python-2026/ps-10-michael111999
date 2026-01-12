import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(path):
    """Loads an image and returns it as a numpy array."""
    # Use the 'path' variable so the test can load its own images
    img = Image.open(path)
    return np.array(img)

def edge_detection(image):
    """Applies Sobel filter and returns a normalized magnitude array."""
    # 1. Convert to grayscale if it's an RGB image
    if len(image.shape) == 3:
        # Standard luminance formula
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    
    # 2. Define Sobel Kernels
    # Vertical changes
    ky = np.array([[1, 2, 1], 
                   [0, 0, 0], 
                   [-1, -2, -1]])
    # Horizontal changes
    kx = np.array([[-1, 0, 1], 
                   [-2, 0, 2], 
                   [-1, 0, 1]])

    # 3. Apply Convolutions
    # 'boundary=symm' is vital for accuracy at the edges of the image
    gx = convolve2d(image, kx, mode='same', boundary='symm')
    gy = convolve2d(image, ky, mode='same', boundary='symm')

    # 4. Calculate Gradient Magnitude
    magnitude = np.sqrt(gx**2 + gy**2)

    # 5. Normalize to 0-255 scale
    # This ensures 'magnitude > 50' in the test works correctly
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude = (magnitude / mag_max) * 255
    
    return magnitude

# Example of how the test uses your code (you don't need to paste this part):
# image = load_image('.tests/lena.jpg')
# edge = edge_detection(image)
# edge_binary = edge > 50
