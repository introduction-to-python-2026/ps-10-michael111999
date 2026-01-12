from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    import numpy as np

    image=Image.open('smurf.jpg')
    image=np.array(image)
    plt.imshow(image)
def edge_detection(image):
    import numpy as np
    from scipy.signal import convolve2d
    gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    plt.imshow(gray_image)

    # Filter for vertical changes (kernelY)
    kernelY = np.array([[1, 2, 1],
                    [ 0,  0,  0],
                    [ -1,  -2,  -1]])

    # Filter for horizontal changes (kernelX)
    kernelX = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
    edges_x = convolve2d(gray_image, kernelX, mode='same')
    edges_y = convolve2d(gray_image, kernelY, mode='same')
    magnitude = np.sqrt(edges_x**2 + edges_y**2)

    # 4. Display the result
    plt.figure(figsize=(10, 5))
    plt.imshow(sobel_final, cmap='gray')
    plt.title('Sobel Edge Detection')
    plt.axis('off')
    plt.show()
    magnitude = (magnitude / magnitude.max()) * 255
    return magnitude
