"""
All exercises from the script.
"""

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.misc
import numpy as np


def basic_cv():
    """Basic openCV usage."""
    mandril = cv2.imread('src/images/mandril.jpg')
    cv2.imshow('Mandril', mandril)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('results/Mandril_copy_cv2.png', mandril)
    print(mandril.shape)
    print(mandril.size)
    print(mandril.dtype)

def basic_mat():
    """Basic matplotlib usage."""
    mandril = plt.imread('src/images/mandril.jpg')
    fig, ax = plt.subplots(1)
    plt.imshow(mandril)
    x = [100, 150, 200, 250]
    y = [50, 100, 150, 200]
    plt.plot(x, y, 'r.', markersize=10)
    rect = Rectangle((50, 50), 50, 100, fill=False, ec='r')
    ax.add_patch(rect)
    plt.axis('off')
    plt.title('Mandril')
    plt.show()
    plt.imsave('results/Mandril_copy_plt', mandril)

def color_cv():
    """Color manipulation with openCV."""
    mandril = cv2.imread('src/images/mandril.jpg')
    mandril_gray = cv2.cvtColor(mandril, cv2.COLOR_BGR2GRAY)
    mandril_hsv = cv2.cvtColor(mandril, cv2.COLOR_BGR2HSV)
    mandril_h = mandril_hsv[:, :, 0]
    mandril_s = mandril_hsv[:, :, 1]
    mandril_v = mandril_hsv[:, :, 2]
    cv2.imshow('Mandril Gray', mandril_gray)
    cv2.imshow('Mandril HSV', mandril_hsv)
    cv2.imshow('Mandril H', mandril_h)
    cv2.imshow('Mandril S', mandril_s)
    cv2.imshow('Mandril V', mandril_v)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def color_mat():
    """Color manipulation with matplotlib."""
    mandril = plt.imread('src/images/mandril.jpg')
    plt.gray()
    plt.imshow(rgb2gray(mandril))
    plt.title('Mandril Gray')
    plt.show()

def rgb2gray(image):
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]

def scaling_cv():
    """Scaling image using cv2."""
    mandril = cv2.imread('src/images/mandril.jpg')
    height, width = mandril.shape[:2]
    scale = 1.75
    big_mandril = cv2.resize(mandril, (int(scale * height), int(scale * width)))
    cv2.imshow('Big Mandril', big_mandril)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def scaling_scipy():
    """Scaling image using scipy."""
    mandril = cv2.imread('src/images/mandril.jpg')
    scale = 0.5
    small_mandril = scipy.misc.imresize(mandril, scale)
    cv2.imshow('Small Mandril', small_mandril)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def operations():
    """Combination of two images."""
    lena = cv2.imread('src/images/lena.png')
    lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    mandril = cv2.imread('src/images/mandril.jpg')
    mandril_gray = cv2.cvtColor(mandril, cv2.COLOR_BGR2GRAY)
    alpha = 0.3
    cv2.imshow('Lena Gray', lena_gray)
    cv2.imshow('Mandril Gray', mandril_gray)
    cv2.imshow('Added Lena Gray and Mandril Gray', lena_gray + mandril_gray)
    cv2.imshow('Subtracted Mandril Gray from Lena Gray', lena_gray - mandril_gray)
    cv2.imshow('Subtracted Lena Gray from Mandril Gray', mandril_gray - lena_gray)
    cv2.imshow('Multiplied Lena Gray with Mandril Gray', lena_gray * mandril_gray)
    cv2.imshow('Linear combination of Lena Gray (' + str(alpha) +
               ') and Mandril Gray (' + str(1 - alpha) + ')',
               linear_combination(lena_gray, mandril_gray, alpha))
    cv2.imshow('Absolute difference (using builtin absdiff)', cv2.absdiff(lena_gray, mandril_gray))
    cv2.imshow('Absolute difference (using my_absdiff)', my_absdiff(lena_gray, mandril_gray))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def linear_combination(img1, img2, alpha):
    return np.uint8(alpha * img1 + (1 - alpha) * img2)

def my_absdiff(img1, img2):
    return np.uint8(abs(img1.astype(float) - img2.astype(float)))

def hist(img):
    h = np.zeros((256, 1), np.float32)
    height, width = img.shape[:2]
    for y in range(height):
        for x in range(width):
            h[img[y][x]] += 1
    return h

def histograms():
    """Several ways to create histograms."""
    mandril = cv2.imread('src/images/mandril.jpg')
    mandril_gray = cv2.cvtColor(mandril, cv2.COLOR_BGR2GRAY)
    mandril_gray_histogram = cv2.calcHist([mandril_gray], [0], None, [256], [0, 256])
    plt.plot(mandril_gray_histogram)
    plt.plot(hist(mandril_gray))
    plt.hist(mandril_gray.ravel(), bins=256)
    plt.show()

def histogram_equalization():
    """Regular and CLAHE equalization."""
    mandril = cv2.imread('src/images/mandril.jpg')
    mandril_gray = cv2.cvtColor(mandril, cv2.COLOR_BGR2GRAY)
    mandril_gray_equalized = cv2.equalizeHist(mandril_gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    mandril_gray_clahed = clahe.apply(mandril_gray)
    cv2.imshow('Regular Mandril Gray', mandril_gray)
    cv2.imshow('Equalized Mandril Gray', mandril_gray_equalized)
    cv2.imshow('CLAHEd Mandril Gray', mandril_gray_clahed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def filtration():
    mandril = cv2.imread('src/images/mandril.jpg')
    mandril_gray = cv2.imread('src/images/mandril.jpg', 0)
    mandril_gaussian_blur = cv2.GaussianBlur(mandril, (5, 5), 0)
    mandril_median_blur = cv2.medianBlur(mandril, 5)
    mandril_bilateral_filter = cv2.bilateralFilter(mandril, 9, 75, 75)
    mandril_gray_sobel_x = cv2.Sobel(mandril_gray, cv2.CV_64F, 1, 0, ksize=3)
    mandril_gray_sobel_y = cv2.Sobel(mandril_gray, cv2.CV_64F, 0, 1, ksize=3)
    mandril_gray_laplacian = cv2.Laplacian(mandril_gray, cv2.CV_64F)
    gabor_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    mandril_gray_gabor = cv2.filter2D(mandril_gray, cv2.CV_8UC3, gabor_kernel)
    cv2.imshow('Mandril', mandril)
    cv2.imshow('Mandril Gaussian Blur', mandril_gaussian_blur)
    cv2.imshow('Mandril Median Blur', mandril_median_blur)
    cv2.imshow('Mandril Bilateral Filter', mandril_bilateral_filter)
    cv2.imshow('Mandril Gray Sobel X', mandril_gray_sobel_x)
    cv2.imshow('Mandril Gray Sobel Y', mandril_gray_sobel_y)
    cv2.imshow('Mandril Gray Laplacian', mandril_gray_laplacian)
    cv2.imshow('Mandril Gray Gabor', mandril_gray_gabor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

filtration()
