import image_utils

from skimage.filters import median
from skimage.morphology import ball


load_image('smurf.jpg')
from skimage.filters import median
from skimage.morphology import ball

clean_image = median(image, ball(3))
edge_detection('smurf.jpg')
                
                                 
                                 
