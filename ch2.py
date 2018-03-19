"""
The 2nd chapter.
"""

import cv2
import numpy as np
from operator import itemgetter

def load_sequence():
    TRESHOLD = 10
    WHITE = (255, 0, 0)
    pedestrian = cv2.imread('src/images/single_pedestrian/in000300.jpg', 0)
    previous_pedestrian = pedestrian
    cv2.imshow('Pedestrian', pedestrian)
    cv2.waitKey(20)
    for i in range(301, 451):
        pedestrian = cv2.imread('src/images/single_pedestrian/in000' + str(i) + '.jpg', 0)
        difference = cv2.absdiff(pedestrian, previous_pedestrian)
        difference_binarized = np.uint8(255 * (difference > TRESHOLD))
        kernel = np.ones((3, 3), np.uint8)
        difference_binarized = cv2.erode(difference_binarized, kernel, iterations=1)
        difference_binarized = cv2.dilate(difference_binarized, kernel, iterations=1)
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(difference_binarized)
        
        if stats.shape[0]:
            print(stats[1:, 4])
            pi = max(enumerate(stats[1:, 4]), key=itemgetter(1))[0] + 1
            cv2.rectangle(difference_binarized, (stats[pi, 0], stats[pi, 1]), (stats[pi, 0] + stats[pi, 2], stats[pi, 1] + stats[pi, 3]), WHITE)
            cv2.putText(difference_binarized, str(stats[pi, 4]), (stats[pi, 0], stats[pi, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE)
            cv2.putText(difference_binarized, str(pi), (np.int(centroids[pi, 0]), np.int(centroids[pi, 1])), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE)


        cv2.imshow('Labels', np.uint8(labels / stats.shape[0] * 255))
        cv2.imshow('Pedestrian', difference_binarized)
        cv2.waitKey(20)
        previous_pedestrian = pedestrian

load_sequence()
