# Do relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

#from scipy import stats

# Read in and grayscale the image
image = mpimg.imread('../images/solidWhiteCurve.jpg')
#image = mpimg.imread('../images/test.jpg')
plt.imshow(image)
plt.show()


color_select = np.copy(image)
rgb_threshold = [100, 100, 100]
# Identify pixels below the threshold
thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])
color_select[thresholds] = [0,0,0]

# Display the image                 
plt.imshow(color_select)
plt.show()


gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray')
plt.show()


roi_left_top = [216, 198]
roi_right_top = [360, 198]
roi_right_bottom = [546, 323]
roi_mid_right_bottom = [420,323]
roi_mid_right_top = [330, 240]
roi_mid_left_top = [246,240]
roi_mid_left_bottom = [156,323]
roi_left_bottom = [30, 323]
vertices = np.array([roi_left_bottom,roi_left_top,
                     roi_right_top,roi_right_bottom, 
                    roi_mid_right_bottom, roi_mid_right_top,
                    roi_mid_left_top, roi_mid_left_bottom], np.int32)

#defining a blank mask to start with
img = np.copy(image)
outer_mask = np.zeros_like(img)   
    
#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
if len(img.shape) > 2:
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
else:
    ignore_mask_color = 255
        
#filling pixels inside the polygon defined by "vertices" with the fill color    
cv2.fillPoly(outer_mask, np.int32([vertices]), ignore_mask_color)
    
#returning the image only where mask pixels are nonzero
masked_image = cv2.bitwise_and(img, outer_mask)
plt.imshow(masked_image)
plt.show()