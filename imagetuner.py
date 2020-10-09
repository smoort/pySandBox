import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils as ut


def callback(x):
    return


def build_image(image, k=5, l=85, u=125):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (k, k), 0)
    canny_image = cv2.Canny(blur_image, l, u)
    masked_image = ut.mask_image(canny_image)
    line_image = ut.hough_image(masked_image, image)
    marked_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    image_line1 = np.concatenate((gray_image, blur_image), axis=1)
    image_line2 = np.concatenate((canny_image, masked_image), axis=1)
    image_line3 = np.concatenate((line_image, marked_image), axis=1)

    plt.imshow(gray_image)
    plt.show()
    plt.imshow(canny_image)
    plt.show()
    plt.imshow(masked_image, cmap='gray')
    plt.title('Output of roi')
    plt.show()
    plt.imshow(line_image)
    plt.title('Output of cleaned lines')
    plt.show()
    plt.imshow(marked_image)
    plt.title('Final image')
    plt.show()

    return marked_image, image_line1, image_line2, image_line3


image = cv2.imread('../images/Challenge4.png')  # read image as grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# print("image shape = ", image.shape)
marked_image, image_line1, image_line2, image_line3 = build_image(image)

"""
cv2.namedWindow('image', cv2.WINDOW_NORMAL) # make a window with name 'image'
cv2.createTrackbar('L', 'image', 85, 255, callback)  # lower threshold trackbar for window 'image
cv2.createTrackbar('U', 'image', 125, 255, callback)  # upper threshold trackbar for window 'image
cv2.createTrackbar('K', 'image', 2, 25, callback) # kernel size
cv2.namedWindow('Final_image', cv2.WINDOW_NORMAL) # make a window with name 'image'


while(1):
    display_image1 = np.concatenate((image_line1, image_line2), axis=0) # to display image side by side
    cv2.imshow('image', display_image1)
    cv2.imshow('Final_image', image_line3)
    key = cv2.waitKey(1) & 0xFF
    if key == 27: #escape key
        break
    l = cv2.getTrackbarPos('L', 'image')
    u = cv2.getTrackbarPos('U', 'image')
    k = cv2.getTrackbarPos('K', 'image')

    k = (k*2) + 1
    marked_image, image_line1, image_line2, image_line3 = build_image(image,k,l,u)

cv2.destroyAllWindows()
"""
