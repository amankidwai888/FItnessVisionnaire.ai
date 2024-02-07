from sklearn.exceptions import DataConversionWarning, InconsistentVersionWarning

import MovenetModule as mm
import SVM as svm
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import os
import pickle
from datetime import datetime
import warnings

# Before loading the SVM model and label encoder
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)

# Load SVM model and label encoder
# Your code to load SVM model and label encoder goes here

# After loading the SVM model and label encoder



#Load SVM model
with open('svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Load label encoder
with open('encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)


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

        # Update frame number
        frame_number += 1

        # Calculate time interval (assuming constant FPS)
        time_interval = frame_number / fps


        # Extract keypoints 5-14
        keypoints_5_to_14 = [mm.get_coordinates(keypoints_with_scores, i)[:2] for i in range(5, 15)]

        # Reset Data_list before appending the new row
        data_list = []

        # Create a row for the DataFrame
        row_data = {'video_name': video_file, 'frame_number': frame_number, 'time_interval': time_interval}
        for i, keypoint in enumerate(keypoints_5_to_14):
            row_data[f'keypoint_{i+5}'] = keypoint

        # Append row data to the list
        data_list.append(row_data)
        if data_list:
            df = pd.DataFrame(data_list)


        #Predict correctness using SVM model

        df_pp=svm.preprocess_new_instance(df)
        df_pp = svm.extract_features_from_instance(df_pp)
        # print(df_pp.shape)
        # print(df_pp.head())

        features = ['video_name', 'frame_number', 'time_interval', 'right_shoulder_x', 'right_shoulder_y',
                    'left_shoulder_x', 'left_shoulder_y', 'right_elbow_x', 'right_elbow_y', 'left_elbow_x',
                    'left_elbow_y', 'right_hand_x', 'right_hand_y', 'left_hand_x', 'left_hand_y', 'waist_right_x',
                    'waist_right_y', 'waist_left_x', 'waist_left_y', 'right_knee_x', 'right_knee_y', 'left_knee_x',
                    'left_knee_y', 'right_elbow_angle', 'left_elbow_angle', 'right_waist_angle', 'left_waist_angle',
                    'bicep_curl_phase_encoded']

        posture = True
        correctness_prediction = svm.predict_correctness(df_pp, svm_model)
        if(correctness_prediction=='correct'):
            posture = True
        elif(correctness_prediction=='incorrect'):
            posture = False

        # #Biceps annotations
        mm.find_angle_and_display(frame, 5, 7, 9, keypoints_with_scores, 0.3, draw=True,correct_posture=posture)
        mm.find_angle_and_display(frame, 6, 8, 10, keypoints_with_scores, 0.3, draw=True,correct_posture=posture)

        # Display correctness prediction for the current frame
        print(f'Frame {frame_number} - Correctness Prediction: {correctness_prediction}')


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