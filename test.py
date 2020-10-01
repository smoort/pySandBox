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


gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray')
plt.show()

# Pull out the x and y sizes and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]

# Define a triangle region of interest 
# Keep in mind the origin (x=0, y=0) is in the upper left in image processing
# Note: if you run this code, you'll find these are not sensible values!!
# But you'll get a chance to play with them soon in a quiz 
left_bottom = [0, 539]
right_bottom = [959, 539]
apex = [472, 300]

# Fit lines (y=Ax+B) to identify the  3 sided region of interest
# np.polyfit() returns the coefficients [A, B] of the fit
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
plt.imshow(blur_gray, cmap='gray')
plt.show()

# Define our parameters for Canny and apply
low_threshold = 20
high_threshold = 80
masked_edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

plt.imshow(masked_edges, cmap='gray')
plt.show()
                                                                                                                                                                                                                                                                                
# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1
theta = np.pi/180
threshold = 20
min_line_length = 50                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
max_line_gap = 30
line_image = np.copy(image)*0 #creating a blank to draw lines on


# Run Hough on edge detected image
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on the blank
for line in lines:
    for x1,y1,x2,y2 in line:
        
#        slope, intercept, r_value, p_value, std_err = stats.linregress([x1,x2],[y1,y2])
        slope = (y2-y1)/(x2-x1)
        print("slope = ", slope)
        if(abs(slope) > 0.5):
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),2)

print("line image shape = ", line_image.shape)
plt.imshow(line_image)
plt.show()

# Define our color criteria
red_threshold = 255
green_threshold = 0
blue_threshold = 0
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Mask for pixels not part of the line
color_thresholds = (line_image[:,:,0] != rgb_threshold[0]) | \
                    (line_image[:,:,1] != rgb_threshold[1]) | \
                    (line_image[:,:,2] != rgb_threshold[2])

# Create a "color" binary image to combine with line image
color_edges = np.dstack((masked_edges, masked_edges, masked_edges)) 

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

region_select = np.copy(color_edges)

print("region threshold shape = ", region_thresholds.shape)

# Color pixels red which are inside the region of interest
region_select[region_thresholds] = [255, 0, 0]

plt.imshow(region_select)
plt.show()

myline_image = np.copy(image)
# Find where image is both colored right and in the region
myline_image[~color_thresholds & region_thresholds] = [255,0,0]
#myline_image[~color_thresholds & ~region_thresholds] = [0,0,0]

plt.imshow(myline_image)
plt.show()

# Draw the lines on the edge image
#combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
#plt.imshow(combo)