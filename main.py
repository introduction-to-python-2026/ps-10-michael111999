import image_utils

from skimage.filters import median
from skimage.morphology import ball

clean_image = median(image, ball(3
edge_detection(clean_image)
threshold = 50 

# Convert to binary (True for edges, False for background)
edge_binary = edgeMAG > threshold
# Display the binary image
plt.imshow(edge_binary, cmap='gray')
plt.title(f"Binary Edge Map (Threshold: {threshold})")
plt.axis('off') # Hide coordinates
plt.show()

# Save as .png
# We multiply by 255 to ensure it saves as a visible white-on-black image
plt.imsave('lena_binary_edges.png', edge_binary.astype(np.uint8) * 255, cmap='gray')


                
                                 
                                 
