"""
The 3rd chapter.
"""

import cv2
import numpy as np

STEP = 10
SLEEP_TIME = 5
TRESHOLD = 10

def frame_subtraction():
    """Object detection using frame subtraction."""
    true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
    with open('src/images/office/temporalROI.txt') as in_file:
        line = in_file.readline()
    roi_start, roi_end = map(int, line.split())
    groundtruth = cv2.imread('src/images/office/groundtruth/gt' + str(roi_start).zfill(6) + '.png')
    office = cv2.imread('src/images/office/input/in' + str(roi_start).zfill(6) + '.jpg', 0)
    previous_office = office
    for i in range(roi_start + 1, roi_end + 1, STEP):
        office = cv2.imread('src/images/office/input/in' + str(i).zfill(6) + '.jpg', 0)
        difference = cv2.absdiff(previous_office, office)
        difference_binarized = np.uint8(255 * (difference > TRESHOLD))
        kernel = np.ones((3, 3), np.uint8)
        difference_binarized = cv2.erode(difference_binarized, kernel, iterations=1)
        difference_binarized = cv2.dilate(difference_binarized, kernel, iterations=1)
        groundtruth = cv2.imread('src/images/office/groundtruth/gt' + str(i).zfill(6) + '.png', 0)
        groundtruth_binarized = np.uint8(255 * (groundtruth == 255))
        true_positive += np.sum(np.logical_and(difference_binarized == 255, groundtruth_binarized == 255))
        true_negative += np.sum(np.logical_and(difference_binarized == 0, groundtruth_binarized == 0))
        false_positive += np.sum(np.logical_and(difference_binarized == 255, groundtruth_binarized == 0))
        false_negative += np.sum(np.logical_and(difference_binarized == 0, groundtruth_binarized == 255))
        cv2.imshow('Office', office)
        cv2.imshow('Groundtruth Binarized', groundtruth_binarized)
        cv2.imshow('Difference Binarized', difference_binarized)
        cv2.waitKey(SLEEP_TIME)
        previous_office = office
    precision = true_positive / (true_positive + false_positive)
    recoil = true_positive / (true_positive + false_negative)
    F1 = 2 * precision * recoil / (precision + recoil)
    print(precision, recoil, F1)

frame_subtraction()
