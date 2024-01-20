import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped[5:]:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)



def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        if((p1>5)&(p2>5)):
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]

            if (c1 > confidence_threshold) & (c2 > confidence_threshold):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

def get_coordinates(keypoints, point_number):
    """
    Get the coordinates of a specific point from the keypoints.

    Parameters:
    - keypoints: Numpy array containing keypoints.
    - point_number: Index of the desired point.

    Returns:
    - Tuple (y, x, confidence): Coordinates and confidence score of the specified point.
    """
    shaped = np.squeeze(np.multiply(keypoints, [480, 640, 1]))

    if point_number < len(shaped):
        y, x, confidence = shaped[point_number]
        return int(y), int(x), confidence
    else:
        return None

import math
import cv2

def find_angle_and_display(img, p1, p2, p3, keypoints,confidence_threshold, draw=True):
    """
    Find the angle between three specified keypoints and display it on the image.

    Parameters:
    - img: Image on which to draw.
    - p1, p2, p3: Indices of the keypoints in the keypoints array.
    - keypoints: Numpy array containing keypoints.
    - draw: Boolean flag to specify whether to draw on the image.

    Returns:
    - angle: Calculated angle in degrees.
    """
    # Get the coordinates of the specified keypoints
    y1, x1, c1 = get_coordinates(keypoints, p1)
    y2, x2, c2 = get_coordinates(keypoints, p2)
    y3, x3, c3 = get_coordinates(keypoints, p3)

    # Calculate the angle
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle = abs(angle)

    # Draw on the image

    if (c1 > confidence_threshold) & (c2 > confidence_threshold) &  (c3 > confidence_threshold):
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 2)

            cv2.circle(img, (x1, y1), 4, (0, 255, 0), -1)
            cv2.circle(img, (x1, y1), 7, (0, 255, 0), 1)
            cv2.circle(img, (x2, y2), 4, (0, 255, 0), -1)
            cv2.circle(img, (x2, y2), 7, (0, 255, 0), 1)
            cv2.circle(img, (x3, y3), 4, (0, 255, 0), -1)
            cv2.circle(img, (x3, y3), 7, (0, 255, 0), 1)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    return angle
