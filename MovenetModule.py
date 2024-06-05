import tensorflow as tf
import numpy as np
# from matplotlib import pyplot as plt
# import cv2
import math


def rescale_points(keypoints,frame):
    """
    Rescale the keypoints coordinates based on the new preview size.

    Parameters:
    - keypoints: Numpy array containing keypoints.
    - width: Original width of the image.
    - height: Original height of the image.
    - context: Context object containing the new preview size.

    Returns:
    - Rescaled keypoints.
    """
    # if(keypoints.ndim==3):
    #     key_y,key_x,c=np.squeeze(keypoints)
    # else:   
    key_y,key_x=np.squeeze(keypoints)



    width = int(1)
    height = int(1)

    new_width, new_height = 1,1
    key_x*= new_width / width
    key_y *= new_height / height

    return int(key_x),int(key_y)


def rescale__three_points(keypoints,frame):
    """
    Rescale the keypoints coordinates based on the new preview size.

    Parameters:
    - keypoints: Numpy array containing keypoints.
    - width: Original width of the image.
    - height: Original height of the image.
    - context: Context object containing the new preview size.

    Returns:
    - Rescaled keypoints.
    """
    # if(keypoints.ndim==3):
    #     key_y,key_x,c=np.squeeze(keypoints)
    # else:   
    key_y,key_x,c=np.squeeze(keypoints)



    width = int(frame.shape[1])
    height = int(frame.shape[0])

    new_width, new_height = 640, 1136
    key_x*= new_width / width
    key_y *= new_height / height

    return int(key_x),int(key_y),int(c)


def find_angle(p1,p2,p3,keypoints):

    y1, x1, c1 = get_coordinates(keypoints, p1)
    y2, x2, c2 = get_coordinates(keypoints, p2)
    y3, x3, c3 = get_coordinates(keypoints, p3)

    # Calculate the angle
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle = abs(angle)
    return angle

    # Draw on the image
    # If there are multiple issues, join them with "and"


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped[5:]:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


def get_squat_body_form(keypoints):
                left_waist_angle = find_angle(11, 12, 13,keypoints)
                right_waist_angle = find_angle(11, 12, 14,keypoints)
                left_knee_angle = find_angle(13, 14, 15,keypoints)
                right_knee_angle = find_angle(14, 13, 16,keypoints)


                if 0 <= left_waist_angle <= 70 or 0 <= right_waist_angle <= 70 and 0 <= left_knee_angle <= 70 or 0 <= right_knee_angle <= 70:
                    return "squat"
                elif 110 <= left_waist_angle <= 180 or 110 <= right_waist_angle <= 180 and 110 <= left_knee_angle <= 180 or 110 <= right_knee_angle <= 180:
                    return "standing"
                else:
                    return "unknown"
                
def get_bicep_body_form(keypoints):
                right_shoulder_angle = find_angle(5, 7, 9, keypoints)
                left_shoulder_angle = find_angle(6, 8, 10, keypoints)

                if 140 <= right_shoulder_angle <= 200 or 140 <= left_shoulder_angle <= 200:
                    return "open"
                elif 0 <= right_shoulder_angle <= 40 or 0 <= left_shoulder_angle <= 40:
                    return "closed"
                else:
                    return "unknown"

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
def display_bicep_count(frame, bicep_count):
    # Get the size of the frame
    height, width, _ = frame.shape

    # Define the text to display
    text = f"Bicep Count: {bicep_count}"
    # Define the font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 1

    # Calculate the size of the text box
    # text_width, text_height = rescale_points((cv2.getTextSize(text, font, font_scale, font_thickness)[0]),frame)
    text_width, text_height =cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    box_width = text_width + 20
    box_height = text_height + 20

    # Calculate the position of the text box
    box_x = width - box_width - 10
    box_y = 10 + text_height
    # Draw the rounded rectangle
    radius = 10
    cv2.rectangle(frame, (box_x, box_y - text_height), (box_x + box_width, box_y + 10), (193, 111, 157), -1)
    cv2.rectangle(frame, (box_x + radius, box_y - text_height), (box_x + box_width - radius, box_y + 10), (193, 111, 157), -1)
    cv2.rectangle(frame, (box_x, box_y - text_height + radius), (box_x + box_width, box_y + 10 - radius), (193, 111, 157), -1)
    cv2.circle(frame, (box_x + radius, box_y - text_height + radius), radius, (193, 111, 157), -1)
    cv2.circle(frame, (box_x + box_width - radius, box_y - text_height + radius), radius, (193, 111, 157), -1)
    cv2.circle(frame, (box_x + radius, box_y + 10 - radius), radius, (193, 111, 157), -1)
    cv2.circle(frame, (box_x + box_width - radius, box_y + 10 - radius), radius, (193, 111, 157), -1)

    # Draw the text
    cv2.putText(frame, text, (box_x + 10, box_y), font, font_scale, (255,255,255), font_thickness)

    return frame

def display_squat_count(frame, squat_count):    
    # Get the size of the frame
    height, width, _ = frame.shape

    # Define the text to display
    text = f"Squat Count: {int(squat_count)}"

    # Define the font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # Calculate the size of the text box
    # text_width, text_height = rescale_points((cv2.getTextSize(text, font, font_scale, font_thickness)[0]),frame)
    text_width, text_height =cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    box_width = text_width + 20
    box_height = text_height + 20

    # Calculate the position of the text box
    box_x = width - box_width - 10
    box_y = 10 + text_height
    # Draw the rounded rectangle
    radius = 10
    cv2.rectangle(frame, (box_x, box_y - text_height), (box_x + box_width, box_y + 10), (193, 111, 157), -1)
    cv2.rectangle(frame, (box_x + radius, box_y - text_height), (box_x + box_width - radius, box_y + 10), (193, 111, 157), -1)
    cv2.rectangle(frame, (box_x, box_y - text_height + radius), (box_x + box_width, box_y + 10 - radius), (193, 111, 157), -1)
    cv2.circle(frame, (box_x + radius, box_y - text_height + radius), radius, (193, 111, 157), -1)
    cv2.circle(frame, (box_x + box_width - radius, box_y - text_height + radius), radius, (193, 111, 157), -1)
    cv2.circle(frame, (box_x + radius, box_y + 10 - radius), radius, (193, 111, 157), -1)
    cv2.circle(frame, (box_x + box_width - radius, box_y + 10 - radius), radius, (193, 111, 157), -1)

    # Draw the text
    cv2.putText(frame, text, (box_x + 10, box_y), font, font_scale,  (255,255,255), font_thickness)

    return frame

def find_angle_and_display(img, p1, p2, p3, keypoints,confidence_threshold, draw=True,correct_posture=1,feedback =""):
    # def find_angle_and_modify_image(img, p1, p2, p3, keypoints, confidence_threshold, draw=True, correct_posture=1, feedback=""):
    # """
    # Find the angle between three specified keypoints and modify the image.

    # Parameters:
    # - img: Image on which to draw.
    # - p1, p2, p3: Indices of the keypoints in the keypoints array.
    # - keypoints: Numpy array containing keypoints.
    # - draw: Boolean flag to specify whether to draw on the image.

    # Returns:
    # - angle: Calculated angle in degrees.
    # - img: Modified image.
    # """
    # Get the coordinates of the specified keypoints
    if correct_posture == 1:
        color = (193, 111, 157)  # Blue color for correct posture
    elif correct_posture == 0:
        color = (0, 0, 255)  # Red color for incorrect posture
    # y1, x1, c1 = rescale__three_points(get_coordinates(keypoints, p1), img)
    # y2, x2, c2 = rescale__three_points(get_coordinates(keypoints, p2), img)
    # y3, x3, c3 = rescale__three_points(get_coordinates(keypoints, p3), img)

    y1, x1, c1 = get_coordinates(keypoints, p1)
    y2, x2, c2 = get_coordinates(keypoints, p2)
    y3, x3, c3 =get_coordinates(keypoints, p3)



    # Calculate the angle
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle = abs(angle)

    # Draw on the image
    # If there are multiple issues, join them with "and"


    if (c1 > confidence_threshold) & (c2 > confidence_threshold) & (c3 > confidence_threshold):
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), color, 2)
            cv2.line(img, (x3, y3), (x2, y2), color, 2)
            cv2.circle(img, (x1, y1), 4, color, -1)
            cv2.circle(img, (x1, y1), 7, color, 1)
            cv2.circle(img, (x2, y2), 4, color, -1)
            cv2.circle(img, (x2, y2), 7, color, 1)
            cv2.circle(img, (x3, y3), 4, color, -1)
            cv2.circle(img, (x3, y3), 7, color, 1)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
            # Display text on the image

            # Display text on the image
            feedback_lines = feedback.split('\n')
            y_position = 420  # Starting y-position for the first line
            if feedback != "":
                # Calculate the size of the box
                text_width, text_height = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_PLAIN, 1.5, 1)[0]
                box_width = 500
                box_height = text_height * len(feedback_lines) + 20

                # Calculate the position of the box
                box_x = 50
                box_y = y_position - box_height

                # Draw the box
                cv2.rectangle(img, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), -1)

                # Draw the text inside the box
                for line in feedback_lines:
                    cv2.putText(img, line, (box_x + 20, box_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                    box_y += 20

    return img

    # """
    # Find the angle between three specified keypoints and display it on the image.

    # Parameters:
    # - img: Image on which to draw.
    # - p1, p2, p3: Indices of the keypoints in the keypoints array.
    # - keypoints: Numpy array containing keypoints.
    # - draw: Boolean flag to specify whether to draw on the image.

    # Returns:
    # - angle: Calculated angle in degrees.
    # """
    # # Get the coordinates of the specified keypoints
    # if correct_posture== 1:
    #     color = (0, 255, 0)  # Green color for correct posture
    # elif correct_posture== 0:
    #     color = (0, 0, 255)  # Red color for incorrect posture

    # y1, x1, c1 = get_coordinates(keypoints, p1)
    # y2, x2, c2 = get_coordinates(keypoints, p2)
    # y3, x3, c3 = get_coordinates(keypoints, p3)

    # # Calculate the angle
    # angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    # if angle < 0:
    #     angle = abs(angle)

    # # Draw on the image
    # # If there are multiple issues, join them with "and"



    # if (c1 > confidence_threshold) & (c2 > confidence_threshold) & (c3 > confidence_threshold):
    #     if draw:
    #         cv2.line(img, (x1, y1), (x2, y2), color, 2)
    #         cv2.line(img, (x3, y3), (x2, y2), color, 2)
    #         cv2.circle(img, (x1, y1), 4, color, -1)
    #         cv2.circle(img, (x1, y1), 7, color, 1)
    #         cv2.circle(img, (x2, y2), 4, color, -1)
    #         cv2.circle(img, (x2, y2), 7, color, 1)
    #         cv2.circle(img, (x3, y3), 4, color, -1)
    #         cv2.circle(img, (x3, y3), 7, color, 1)
    #         cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2,color, 2)
    #         # # Display text on the image
    #         feedback_lines = feedback.split('\n')
    #         y_position = 50  # Starting y-position for the first line

    #         for line in feedback_lines:
    #             cv2.putText(img, line, (50, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #             y_position += 30  # Increment y-position for the next line

    # return angle