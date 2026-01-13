import image_utils

from skimage.filters import median
from skimage.morphology import ball

clean_image = median(image, ball(3


edgeMAG = edge_detection(clean_image)
threshold_value = 50 

# B. Convert to binary (True for edges, False for background)
edge_binary = edgeMAG > threshold_value

# C. Display the binary image to check your work
plt.imshow(edge_binary, cmap='gray')
plt.title("Binary Edge Map")
plt.axis('off')
plt.show()

# D. Save the image as a .png file
# We multiply by 255 to ensure 'True' becomes 'White' (255)
plt.imsave('my_binary_edges.png', edge_binary.astype(np.uint8) * 255, cmap='gray')

                
                                 
                                 
