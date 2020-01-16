import math
import cv2

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
    mask = np.zeros(img)
    
    if(img.shape[1] >2):
        channel_count = img.shape[2]
        ignor_mask_color = (255,) * channel_count
    else:
        ignor_mask_color = 255

    cv2.fillPoly(mask, vertices, ignor_mask_color)
    masked_img = cv2.bitwise_and(img, mask)

    return masked_img


