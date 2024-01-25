import MovenetModule as mm
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import os
from datetime import datetime


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

# Initialize DataFrame
columns = ['video_name', 'frame_number', 'time_interval'] + [f'keypoint_{i}' for i in range(5, 15)]
df = pd.DataFrame(columns=columns)


# Make Detections

# # code for training annotations on dataset
folder_path = 'Train'
output_folder_path = 'Train_annotated'


# Get a list of all video files in the folder
video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

# Iterate over each video file
for video_file in video_files:
    video_path = os.path.join(folder_path, video_file)
    output_video_path = os.path.join(output_folder_path, f"annotated_{video_file}")
    cap = cv2.VideoCapture(video_path)

    #for reading from webcam
    # cap = cv2.VideoCapture(0)

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))


    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()


        if not ret:
            print("Error reading frame. Exiting...")
            break

        if not frame.size:
            print("Empty frame. Skipping...")
            continue

        frame = cv2.resize(frame, (640,480))

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
        # keypoints_with_scores[:, :, 0] *= frame_width  # x-coordinates
        # keypoints_with_scores[:, :, 1] *= frame_height  # y-coordinates
        # print(frame_height)
        # print(frame_width)


        # Update frame number
        frame_number += 1

        # Calculate time interval (assuming constant FPS)
        time_interval = frame_number / fps

        # Extract keypoints 5-14
        keypoints_5_to_14 = [keypoints_with_scores[0, i, :2].tolist() for i in range(5, 15)]

        # Create a row for the DataFrame
        row_data = {'video_name': video_file, 'frame_number': frame_number, 'time_interval': time_interval}
        for i, keypoint in enumerate(keypoints_5_to_14):
            row_data[f'keypoint_{i+5}'] = keypoint

        # Append row to DataFrame
        df = df.append(row_data, ignore_index=True)


        # #Biceps annotations
        mm.find_angle_and_display(frame, 5, 7, 9, keypoints_with_scores,0.3, draw=True)
        mm.find_angle_and_display(frame, 6, 8, 10, keypoints_with_scores,0.3, draw=True)

        # Save the annotated frame to the output video
        out.write(frame)

        cv2.imshow('MoveNet Lightning', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Save DataFrame to CSV
csv_file_path = 'keypoints_data.csv'
df.to_csv(csv_file_path, index=False)

print(f'DataFrame saved to {csv_file_path}')