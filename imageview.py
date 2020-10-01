import cv2
import numpy as np 
import matplotlib.pyplot as plt


def callback(x):
    print(x)

def resize(img):
    scale_percent = 60 # percent of original size
    new_width = int(img.shape[1] * scale_percent / 100)
    new_height = int(img.shape[0] * scale_percent / 100)
    dim = (new_width, new_height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
img = cv2.imread('../images/exit-ramp.jpg') #read image as grayscale

# Convert image to gray scale
gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
canny_image = cv2.Canny(blur_image, 85, 255) 

"""
roi_left_top = [100, 198]
roi_right_top = [360, 198]
roi_right_bottom = [546, 323]
roi_mid_right_bottom = [420,323]
roi_mid_right_top = [330, 240]
roi_mid_left_top = [246,240]
roi_mid_left_bottom = [156,323]
roi_left_bottom = [30, 323]
"""
roi_left_top = [360, 330]
roi_right_top = [600, 330]
roi_right_bottom = [910, 539]
roi_mid_right_bottom = [700,539]
roi_mid_right_top = [550, 400]
roi_mid_left_top = [410,400]
roi_mid_left_bottom = [260,539]
roi_left_bottom = [50, 539]
vertices = np.array([roi_left_bottom,roi_left_top,
                     roi_right_top,roi_right_bottom, 
                    roi_mid_right_bottom, roi_mid_right_top,
                    roi_mid_left_top, roi_mid_left_bottom], np.int32)
#defining a blank mask to start with
masked_image = np.copy(canny_image)
mask = np.zeros_like(masked_image)   
ignore_mask_color = 255
#filling pixels inside the polygon defined by "vertices" with the fill color    
cv2.fillPoly(mask, np.int32([vertices]), ignore_mask_color)
#returning the image only where mask pixels are nonzero
masked_image = cv2.bitwise_and(canny_image, mask)

rho = 1
theta = np.pi/180
threshold = 20
min_line_len = 50                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
max_line_gap = 30
lines = cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
#line_img = np.zeros((gray_image.shape[0], img.shape[1], 3), dtype=np.uint8)
line_img = np.zeros_like(gray_image)   
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_img, (x1, y1), (x2, y2), [255, 0, 0], 2)
plt.imshow(line_img, cmap='gray')
plt.show()

cv2.namedWindow('image') # make a window with name 'image'
cv2.createTrackbar('L', 'image', 85, 255, callback) #lower threshold trackbar for window 'image
cv2.createTrackbar('U', 'image', 125, 255, callback) #upper threshold trackbar for window 'image
cv2.createTrackbar('K', 'image', 2, 25, callback) #kernel size

while(1):
    image_line1 = np.concatenate((resize(gray_image), resize(blur_image)), axis=1) # to display image side by side
    image_line2 = np.concatenate((resize(canny_image), resize(line_img)), axis=1) # to display image side by side
#    image_line3 = np.concatenate((line_img, masked_image), axis=1) # to display image side by side
    display_image = np.concatenate((image_line1, image_line2), axis=0) # to display image side by side
    cv2.imshow('image', display_image)
    k = cv2.waitKey(1) & 0xFF
    if k == 27: #escape key
        break
    l = cv2.getTrackbarPos('L', 'image')
    u = cv2.getTrackbarPos('U', 'image')
    k = cv2.getTrackbarPos('K', 'image')

    k = (k*2) + 1    
    blur_image = cv2.GaussianBlur(gray_image, (k, k), 0)
    canny_image = cv2.Canny(blur_image, l, u)
    masked_image = cv2.bitwise_and(canny_image, mask)
    lines = cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros_like(gray_image)   
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), [255, 0, 0], 2)
    
cv2.destroyAllWindows()