#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 22:43:10 2020

@author: ms
"""
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt


def chk_and_get_slope_intercept(x1, y1, x2, y2):
    """
    Finds slope of all lines detected by hough transform to determine if a line is vertical or horizontal.

    Returns True if line is vertical and False if line is horizontal
    """

    lane_side = "invalid"
    slope = 0
    intercept = 0

    transformed_y1 = 799 - y1
    transformed_y2 = 799 - y2

    if(transformed_y1 > transformed_y2):
        top_x = x1
        top_y = transformed_y1
        bottom_x = x2
        bottom_y = transformed_y2
    else:
        top_x = x2
        top_y = transformed_y2
        bottom_x = x1
        bottom_y = transformed_y1
    
    if (top_x != bottom_x):
        slope = (top_y-bottom_y)/(top_x-bottom_x)
    
        
    if(abs(slope) > 0.5):

        if(bottom_x <= 1280/2):
            if(slope > 0):
                intercept = bottom_y - (slope * bottom_x)                
                lane_side = 'left'
        else:
            if(slope < 0):
                intercept = bottom_y - (slope * bottom_x)
                lane_side = 'right'

                
    return lane_side, slope, intercept

def remove_outliers(lines, slope_avg, intercept_avg):
    
    clines = []
    distance_from_mean = []
    for line in lines:
        clines.append([line[0]])
    
    """
    
    for line in lines:
        dist = math.sqrt((line[1] - slope_avg)**2 + 
                         ((line[2] - intercept_avg)**2))
        distance_from_mean.append(dist)
        clines.append([line[0]])
    
#    print('distance from mean', distance_from_mean)
#    print('Avg distance from mean = ', sum(distance_from_mean)/len(distance_from_mean))

    
    z = np.abs(stats.zscore(distance_from_mean))
    print('z = ', z)
    print(np.where(z > 1))
    print(np.array(clines)[np.where(z > 1)])
    
    sorted_distance_from_mean_sum = 0
    sorted_distance_from_mean = np.sort(distance_from_mean)
    half_number_of_points = round(len(sorted_distance_from_mean)/2)
    for i in range(half_number_of_points):
        sorted_distance_from_mean_sum += sorted_distance_from_mean[i]
    sorted_distance_from_mean_avg = sorted_distance_from_mean_sum/half_number_of_points
    outer_limit_distance = sorted_distance_from_mean_avg * 2
    print("Avg of first half points = ", sorted_distance_from_mean_avg)
    print("Outer limit distance = ", outer_limit_distance)
    print("***** Ignored Line Start*****")
    print(np.where(distance_from_mean > outer_limit_distance))
    print(np.array(clines)[np.where(distance_from_mean > outer_limit_distance)])
    print("***** Ignored Line End *****")
        
    return np.array(clines)[np.where(distance_from_mean < outer_limit_distance)]
    """
    return np.array(clines)

def cleanup_lines(lines):
    """
    Do the following clean-up
    
    1. Remove horizontal lines
    2. Remove any outlier lines that are not part of the lane
    
    """
    valid_lines = []
    left_lane_lines = []
    right_lane_lines = []
    
    left_slope_sum = 0
    left_intercept_sum = 0
    left_slope_avg = 0
    left_intercept_avg = 0
    right_slope_sum = 0
    right_intercept_sum = 0
    right_slope_avg = 0
    right_intercept_avg = 0

#    print("Total number of line = ", len(lines))
#    print("***** Invalid Lines Start *****")
    for line in lines:
#        print('original line type is:', type(line))
#        print('original line shape is:', line.shape)
#        print('original line = ', line)
        
        for x1,y1,x2,y2 in line:
            lane_side, slope, intercept = chk_and_get_slope_intercept(x1, y1, x2, y2)
#            print('lane side = ', lane_side)
            if lane_side != 'invalid':
                
#                print(x1,y1,x2,y2)
#                print('lane side = ', lane_side)
#                print('slope = ', slope)
#                print('intercept = ', intercept)

                if lane_side == 'left':
                    left_lane_lines.append(((x1, y1, x2, y2), slope, intercept))
                    left_slope_sum += slope
                    left_intercept_sum += intercept
#                    print("left slope sum = ", left_slope_sum)
#                    print("left intercept sum = ", left_intercept_sum)
                else:
                    right_lane_lines.append(((x1, y1, x2, y2), slope, intercept))
                    right_slope_sum += slope
                    right_intercept_sum += intercept
#                    print("right slope sum = ", right_slope_sum)
#                    print("right intercept sum = ", right_intercept_sum)
#            else:
#                print(x1,y1,x2,y2)
#    print("***** Invalid Lines End *****")
                    
    if len(left_lane_lines) > 0:
        left_slope_avg = left_slope_sum / len(left_lane_lines)
        left_intercept_avg = left_intercept_sum / len(left_lane_lines)
    if len(right_lane_lines) > 0:
        right_slope_avg = right_slope_sum / len(right_lane_lines)
        right_intercept_avg = right_intercept_sum / len(right_lane_lines)
        

#    print("No. of left lane line = ", len(left_lane_lines))
#    print("Accumulated left slope, intecept = ", left_slope_sum, " ", left_intercept_sum)
#    print("left avg slope and intercept = ", left_slope_avg, " ", left_intercept_avg)
#    print("No. of right lane line = ", len(right_lane_lines))
#    print("Accumulated right slope, intecept = ", right_slope_sum, " ", right_intercept_sum)
#    print("right avg slope and intercept = ", right_slope_avg, " ", right_intercept_avg)

    
    cleaned_left_lane_lines = remove_outliers(left_lane_lines, left_slope_avg, left_intercept_avg)
    cleaned_right_lane_lines = remove_outliers(right_lane_lines, right_slope_avg, right_intercept_avg)
#    print('left line type = ', cleaned_left_lane_lines.shape)
#    print('right line shape = ', cleaned_right_lane_lines.shape)
#    valid_lines = np.concatenate((cleaned_left_lane_lines, cleaned_right_lane_lines))

    if (len(left_lane_lines) != 0 and len(right_lane_lines) != 0):
        valid_lines = np.concatenate((cleaned_left_lane_lines, cleaned_right_lane_lines))
    elif (len(left_lane_lines) == 0 and len(right_lane_lines) != 0):
        valid_lines = cleaned_right_lane_lines
    elif (len(left_lane_lines) != 0 and len(right_lane_lines) == 0):
        valid_lines = cleaned_left_lane_lines

    return valid_lines, left_slope_avg, left_intercept_avg, right_slope_avg, right_intercept_avg

def resize(image):
    scale_percent = 40 # percent of original size
    new_width = int(image.shape[1] * scale_percent / 100)
    new_height = int(image.shape[0] * scale_percent / 100)
    dim = (new_width, new_height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

def mask_image(canny_image):
    
    """
    roi_left_top = [410, 330]
    roi_right_top = [550, 330]
    roi_right_bottom = [940, 539]
    roi_left_bottom = [130, 539]

    vertices = np.array([roi_left_bottom,roi_left_top,roi_right_top,roi_right_bottom], np.int32)

    roi_file = "roi1.jpg"

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
    
    """
    
    roi_left_top = [550, 480]
    roi_right_top = [750, 480]
    roi_right_bottom = [1100, 710]
    roi_mid_right_bottom = [1000,710]
    roi_mid_right_top = [750, 550]
    roi_mid_left_top = [550,550]
    roi_mid_left_bottom = [350,710]
    roi_left_bottom = [250, 710]
    vertices = np.array([roi_left_bottom,roi_left_top,
                     roi_right_top,roi_right_bottom, 
                    roi_mid_right_bottom, roi_mid_right_top,
                    roi_mid_left_top, roi_mid_left_bottom], np.int32)
    
    ignore_mask_color = 255

    #defining a blank mask to start with    
    mask = np.zeros_like(canny_image) 
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, np.int32([vertices]), ignore_mask_color)
    #returning the image only where mask pixels are nonzero
#    plt.imshow(mask)
#    plt.show()
    return cv2.bitwise_and(canny_image, mask)
    
def hough_image(masked_image, original_image):
    rho = 1
    theta = np.pi/180
    threshold = 20
    min_line_len = 50                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    max_line_gap = 30
    lines = cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
#    print("lines type = ", type(lines))
    
    """
    hough_line_image = np.zeros_like(original_image)
    for i, line in enumerate(lines):
        for x1,y1,x2,y2 in line:
            print("line ",i," = ",x1,y1,x2,y2)
            cv2.line(hough_line_image, (x1, y1), (x2, y2), [255, 0, 0], 2)
            plt.imshow(hough_line_image)
            plt.annotate(i,xy=((x1+x2)/2,(y1+y2)/2),xycoords = 'data',color='white', fontsize=5)
            plt.show()
    
    plt.imshow(hough_line_image)
    plt.title('Original output of Hough before cleanup')
    for i, line in enumerate(lines):
        for x1,y1,x2,y2 in line:
            print("line ",i," = ",x1,y1,x2,y2)
            plt.annotate(i,xy=((x1+x2)/2,(y1+y2)/2),xycoords = 'data',color='white', fontsize=10)
    plt.show()
    """
    
    line_image = np.zeros_like(original_image)
    if lines is not None:
        valid_lines, left_slope_avg, left_intercept_avg, right_slope_avg, right_intercept_avg = cleanup_lines(lines)
#    print("***** Valid Points Start *****")
        for line in valid_lines:
            for x1,y1,x2,y2 in line:
#                print(x1,y1,x2,y2)
                cv2.line(line_image, (x1, y1), (x2, y2), [255, 0, 0], 2)
#    print("***** Valid Points End *****")
    return line_image