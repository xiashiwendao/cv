import math
import cv2
from PIL import Image
import numpy as np
from matplotlib import image as mpimg
from matplotlib import pyplot as plt

def grayscale(img):
    '''灰度变换'''
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    '''canny边缘检测'''
    return cv2.canny(low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    '''高斯去噪'''
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def median_blur(img, kernel_size):
    '''中值去噪'''
    return cv2.medianBlur(img, kernel_size)

def region_of_interest(img, vertices):
    '''标注出来感兴趣的区域，边缘检测只是关注感兴趣的图像区域'''
    mask = np.zeros_like(img)
    
    if(img.shape[1] >2):
        channel_count = img.shape[2]
        ignor_mask_color = (255,) * channel_count
    else:
        ignor_mask_color = 255

    cv2.fillPoly(mask, vertices, ignor_mask_color)
    masked_img = cv2.bitwise_and(img, mask)

    return masked_img

def test_masked_img():
    img = mpimg.imread('dataset\\edge_detect\\edgeDetect.jpg')
    s = img.shape
    print(img.shape)
    vertices = np.array([[(0, s[0]), (162, 105), (207, 105), (s[1], s[0])]], dtype=np.int32)
    img_mask = region_of_interest(img, vertices)

    return img_mask

def draw_line(img, lines, color=[255, 0, 0], thickness=2):
    '''
    画线段
    '''
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def get_y_intercept(lane_lines, slopes):
    '''
    根据输入的线段和斜率，计算y方向的截距和平均斜率；输入的线段是交通线的虚线，边缘检测的时候要标注成实线
    '''
    slopes = slopes[~np.isnan(slopes)]
    slopes = slopes[~np.isinf(slopes)]
    avg_slopes = np.mean(slopes)

    lane_lines = lane_lines.reshape(slopes.shape[0] * 2, slopes.shape[1]//2)
    x_mean, y_mean = np.mean(reshape_slope, axis=0)

    return y_mean - (x_mean * avg_slopes), avg_slopes

def get_x_intercept(y1, y2, slope, b):
    '''
    根据y1, y2起点终点y值，以及斜率slope和截距（根据get_y_intercept函数获得）来计算起点和终点x值
    '''
    if not (~np.isnan(slope) and ~np.isnan(b)):
        x_1 = x_2 = 0
    else:
        x_1 = (y_1 - b) / slope
        x_2 = (y_2 - b) / slope

    return x_1, x_2

prev_left_x1 = 0
prev_left_x2 = 0
prev_right_x1 = 0
prev_right_x2 = 0
prev_left_avg_m = 0
prev_right_avg_m = 0
prev_left_b = 0
prev_right_b = 0
prev_left_line = 0
prev_right_line = 0

def draw_lines_extrapolated(img, lines, color=[255, 0, 0], thickness=10):
    imgshape = img.shape
    lines = lines.reshape((lines.shape[0], lines.shape[2]))
    y_min = lines.reshape((lines.shape[0] * 2, lines.shape[1]//2))[:, 1].min()

    slopes = (lines[:3] - lines[:,1])/(lines[:,2] - lines[:,0])
    slopes = slopes[~np.isinf(slopes)]
    slopes = slopes[~np.isnan(slopes)]
    # 0.5是经验值
    left_lines = lines[slopes < -0.5]
    right_lines = lines[slopes > 0.5]
    left_slope = slopes[slopes < -0.5]
    right_slope = slopes[slopes > 0.5]

    global prev_left_avg_m
    global prev_right_avg_m
    global prev_left_b
    global prev_right_b

    left_b, left_avg_m = get_y_intercept(left_lines, left_slope)
    right_b, right_avg_m = get_y_intercept(right_lines, right_slope)

    keep_prev_left = False
    keep_prev_right = False

    if left_avg_m < -0.83 or left_avg_m > -0.36:
        left_avg_m = prev_left_avg_m
        left_b = prev_left_b
        keep_prev_left = True
    if right_avg_m > 0.83 or right_avg_m < 0.36:
        right_avg_m = prev_right_avg_m
        right_b = prev_right_b
        keep_prev_right = True

    prev_left_avg_m = left_avg_m
    prev_right_avg_m = right_avg_m
    prev_left_b = left_b
    prev_right_b = right_b

    left_x1, left_x2 = get_x_intercept(y_1 = y_min, y2 = imgshape[0], slope=left_avg_m, b=left_b)
    right_x1, right_x2 = get_x_intercept(y_1=y_min, y2=imgshape[0], slope=right_avg_m, b=right_b)





img_mask = test_masked_img()
plt.imshow(img_mask)
plt.show()