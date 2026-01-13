import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import ball
from image_utils import load_image, edge_detection

# Step 1: Load the original image
# Use the exact filename in your repository
image = load_image('smurf.jpg')

# Step 2: Perform noise suppression
# ball(3) creates a disk-like neighborhood for the median filter
clean_image = median(image, ball(3))

# Step 3: Detect edges (returns edgeMAG)
edgeMAG = edge_detection(clean_image)

# Step 4: Convert to binary and save
# Choosing a threshold (50 is usually recommended for Lena)
threshold = 50
edge_binary = edgeMAG > threshold

# Save the resulting edge-detected image as a .png
# Multiplying by 255 ensures the binary image is visible white-on-black
plt.imsave('smurf_edges_detected.png', edge_binary.astype(np.uint8) * 255, cmap='gray')

# Optional: Display the result
plt.imshow(edge_binary, cmap='gray')
plt.axis('off')
plt.show()
                                 
