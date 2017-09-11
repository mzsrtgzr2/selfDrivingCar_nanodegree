import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image
image = mpimg.imread('test.jpg')

# Grab the x and y sizes and make two copies of the image
# With one copy we'll extract only the pixels that meet our selection,
# then we'll paint those pixels red in the original image to see our selection
# overlaid on the original.
ysize = image.shape[0]
xsize = image.shape[1]
cropped_image = np.copy(image)


# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
R = 100
region_thresholds = ((YY-ysize/2)**2 + (XX-xsize/2)**2)<R**2

# Display our two output images
#plt.imshow(color_select)
plt.imshow(region_thresholds)
plt.show()

input("Press Enter to continue...")
