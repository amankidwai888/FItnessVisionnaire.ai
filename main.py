import MovenetModule as mm
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
import os


# load model
interpreter = tf.lite.Interpreter(model_path='movenet_singlepose_lightning.tflite')
interpreter.allocate_tensors()

# Draw Edges
EDGES = {
    # (0, 1): 'm',
    # (0, 2): 'c',
    # (1, 3): 'm',
    # (2, 4): 'c',
    # (0, 5): 'm',
    # (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

shaped = np.squeeze(
    np.multiply(interpreter.get_tensor(interpreter.get_output_details()[0]['index']), [480, 640, 1]))
for kp in shaped:
    ky, kx, kp_conf = kp
    # print(int(ky), int(kx), kp_conf)

for edge, color in EDGES.items():
    p1, p2 = edge
    y1, x1, c1 = shaped[p1]
    y2, x2, c2 = shaped[p2]
    # print((int(x2), int(y2)))



# Make Detections

# # code for training annotations on dataset
folder_path = 'Train'
output_folder_path = 'Train'


# Get a list of all video files in the folder
video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

# Iterate over each video file
for video_file in video_files:
    video_path = os.path.join(folder_path, video_file)
    output_video_path = os.path.join(output_folder_path, f"annotated_{video_file}")
    # cap = cv2.VideoCapture(video_path)

    #for reading from webcam
    cap = cv2.VideoCapture(0)

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


    while cap.isOpened():
        ret, frame = cap.read()


        # Reshape image
        img = frame.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
        input_image = tf.cast(img, dtype=tf.float32)

        # Setup input and output
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()


        # Make predictions
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

        # Adjust keypoints based on the original image size
        keypoints_with_scores[:, :, 0] *= frame_width / 192  # x-coordinates
        keypoints_with_scores[:, :, 1] *= frame_height / 192  # y-coordinates

        # Rendering
        # mm.draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
        # mm.draw_keypoints(frame, keypoints_with_scores, 0.4)

        #Biceps annotations
        mm.find_angle_and_display(frame, 5, 7, 9, keypoints_with_scores,0.3, draw=True)
        mm.find_angle_and_display(frame, 6, 8, 10, keypoints_with_scores,0.3, draw=True)

        # Save the annotated frame to the output video
        # out.write(frame)

        cv2.imshow('MoveNet Lightning', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

