from hmac import new

import pandas as pd
import pickle
import math
import warnings
from sklearn.exceptions import InconsistentVersionWarning, DataConversionWarning

# Before loading the SVM model and label encoder
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)

from sklearn.preprocessing import LabelEncoder

# Assuming svm_classifier is your trained SVM classifier
# Define a function to split coordinates and create new columns
# Define a function to split coordinates and create new columns

def split_coordinates(df, column_name):
    new_columns = ['{}_x'.format(column_name), '{}_y'.format(column_name)]

    # Check if the column contains tuples, and handle accordingly
    if df[column_name].apply(lambda x: isinstance(x, tuple)).all():
        df[new_columns] = pd.DataFrame(df[column_name].tolist(), columns=new_columns)
    else:
        # Create new columns if they don't exist
        if new_columns[0] not in df.columns:
            df[new_columns] = df[column_name].apply(lambda x: pd.Series(x.split(' ')))

    df.drop(column_name, axis=1, inplace=True)


# Define a function to remove brackets and commas
import numpy as np
import pandas as pd
import tensorflow as tf
import math

# Define the RBF kernel function
def rbf_kernel(x1, x2, gamma=0.1):
    return tf.exp(-gamma * tf.reduce_sum((x1 - x2) ** 2, axis=-1))

# Define the non-linear SVM model
class NonLinearSVM(tf.keras.Model):
    def __init__(self, gamma=0.1):
        super(NonLinearSVM, self).__init__()
        self.gamma = gamma

    def call(self, inputs):
        support_vectors = self.support_vectors
        dual_coefficients = self.dual_coefficients
        kernel_matrix = rbf_kernel(inputs, support_vectors, self.gamma)
        return tf.reduce_sum(tf.multiply(kernel_matrix, dual_coefficients), axis=-1)

# Define the loss function for SVM
def hinge_loss(y_true, y_pred):
    return tf.reduce_mean(tf.maximum(0., 1 - y_true * y_pred))

# Function to calculate Euclidean distance
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to calculate cumulative size
def calculate_cumulative_size(df):
    cumulative_size = []
    cumulative_distance = 0

    for index, row in df.iterrows():
        shoulder_waist_dist = euclidean_distance(row['right_shoulder_x'], row['right_shoulder_y'], row['waist_right_x'], row['waist_right_y'])
        waist_knees_dist = euclidean_distance(row['waist_right_x'], row['waist_right_y'], row['right_knee_x'], row['right_knee_y'])
        knees_feet_dist = euclidean_distance(row['right_knee_x'], row['right_knee_y'], row['right_knee_x'], row['right_knee_y'])  # Assuming feet are at the knees position

        cumulative_distance += shoulder_waist_dist + waist_knees_dist + knees_feet_dist
        cumulative_size.append(cumulative_distance)

    return cumulative_size

# Function to split coordinates and create new columns
def split_coordinates(df, column_name):
    new_columns = ['{}_x'.format(column_name), '{}_y'.format(column_name)]
    df[new_columns] = df[column_name].apply(lambda x: pd.Series(x.split(' ')))
    df.drop(column_name, axis=1, inplace=True)

# Function to determine orientation
def determine_orientation(row):
    # Get the x-coordinates of relevant keypoints
    right_shoulder_x = row['right_shoulder_x']
    left_shoulder_x = row['left_shoulder_x']
    right_waist_x = row['waist_right_x']
    left_waist_x = row['waist_left_x']
    right_knee_x = row['right_knee_x']
    left_knee_x = row['left_knee_x']

    # Calculate the average x-coordinate of shoulders
    avg_shoulder_x = (right_shoulder_x + left_shoulder_x) / 2

    # Calculate the average x-coordinate of waists
    avg_waist_x = (right_waist_x + left_waist_x) / 2

    # Calculate the average x-coordinate of knees
    avg_knee_x = (right_knee_x + left_knee_x) / 2

    # Check if the person is front-facing
    if abs(avg_shoulder_x - avg_waist_x) < 20 and abs(avg_waist_x - avg_knee_x) < 20:
        return 'Front-Facing'

    # Check if the person is right-facing
    elif avg_shoulder_x > avg_waist_x and avg_waist_x > avg_knee_x:
        return 'Right-Facing'

    # Check if the person is left-facing
    elif avg_shoulder_x < avg_waist_x and avg_waist_x < avg_knee_x:
        return 'Left-Facing'

    # If orientation cannot be determined
    else:
        return 'Unknown'

# Function to remove brackets and commas from coordinates
def clean_coordinates(df, column_name):
    for coordinate_type in ['_x', '_y']:
        df[column_name + coordinate_type] = df[column_name + coordinate_type].str.replace(r'[()]', '')
        df[column_name + coordinate_type] = df[column_name + coordinate_type].str.replace(',', '')

# Function to find angle
def find_angle(row, shoulder, elbow, hand):
    x1, y1 = row[f'{shoulder}_x'], row[f'{shoulder}_y']
    x2, y2 = row[f'{elbow}_x'], row[f'{elbow}_y']
    x3, y3 = row[f'{hand}_x'], row[f'{hand}_y']

    # Calculate the Angle
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # Adjust the angle to be in the range [0, 360)
    if angle < 0:
        angle += 360

    # Check if the angle is in the [180, 360) range, then subtract it from 360
    if 180 <= angle < 360:
        angle = 360 - angle

    return angle


def preprocess_new_instance(new_instance, scaler=None, encoder=None):
    # Assume new_instance is a DataFrame with the same structure as your training data
    column_mapping = {
        'keypoint_3': 'right_ear',
        'keypoint_4': 'left_ear',
        'keypoint_5': 'right_shoulder',
        'keypoint_6': 'left_shoulder',
        'keypoint_7': 'right_elbow',
        'keypoint_8': 'left_elbow',
        'keypoint_9': 'right_hand',
        'keypoint_10': 'left_hand',
        'keypoint_11': 'waist_right',
        'keypoint_12': 'waist_left',
        'keypoint_13': 'right_knee',
        'keypoint_14': 'left_knee',
        'keypoint_15': 'right_foot',
        'keypoint_16': 'left_foot'
    }

    # Apply the same column renaming logic if necessary
    new_instance.rename(columns=column_mapping, inplace=True)

    keypoint_columns = ['right_ear', 'left_ear', 'right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow',
                        'right_hand', 'left_hand', 'waist_right', 'waist_left',
                        'right_knee', 'left_knee', 'right_foot', 'left_foot']
    for column in keypoint_columns:
        split_coordinates(new_instance, column)

    for column in keypoint_columns:
        clean_coordinates(new_instance, column)



    df_copy = pd.DataFrame(columns=["video_name"])
    df_copy['video_name'] = new_instance["video_name"]

    new_instance = new_instance.apply(pd.to_numeric, errors='coerce')

    # Calculate imaginary feet points based on knee position
    new_instance['right_foot_x'] = np.where(new_instance['right_foot_x'].isnull(), new_instance['right_knee_x'],
                                            new_instance['right_foot_x'])  # Assuming feet position is same as knee
    new_instance['right_foot_y'] = np.where(new_instance['right_foot_y'].isnull(), new_instance['right_knee_y'] + 100,
                                            new_instance['right_foot_y'])  # Assuming feet position is below knee
    new_instance['left_foot_x'] = np.where(new_instance['left_foot_x'].isnull(), new_instance['left_knee_x'],
                                           new_instance['left_foot_x'])  # Assuming feet position is same as knee
    new_instance['left_foot_y'] = np.where(new_instance['left_foot_y'].isnull(), new_instance['left_knee_y'] + 100,
                                           new_instance['left_foot_y'])  # Assuming feet position is below knee

    # Calculate imaginary knee points based on waist position
    new_instance['right_knee_x'] = np.where(new_instance['right_knee_x'].isnull(), new_instance['waist_right_x'] + 30,
                                            new_instance[
                                                'right_knee_x'])  # Assuming a typical distance from waist to knee
    new_instance['right_knee_y'] = np.where(new_instance['right_knee_y'].isnull(), new_instance['waist_right_y'] + 50,
                                            new_instance[
                                                'right_knee_y'])  # Assuming a typical distance from waist to knee
    new_instance['left_knee_x'] = np.where(new_instance['left_knee_x'].isnull(), new_instance['waist_left_x'] - 30,
                                           new_instance[
                                               'left_knee_x'])  # Assuming a typical distance from waist to knee
    new_instance['left_knee_y'] = np.where(new_instance['left_knee_y'].isnull(), new_instance['waist_left_y'] + 50,
                                           new_instance[
                                               'left_knee_y'])  # Assuming a typical distance from waist to knee

    # Add cumulative size column to the DataFrame
    new_instance['cumulative_size'] = calculate_cumulative_size(new_instance)

    # Normalize keypoints based on the cumulative body size
    keypoints_columns = new_instance.columns[3:-1]  # Extract columns containing keypoints
    for column in keypoints_columns:
        new_instance[column] = new_instance[column] / new_instance['cumulative_size']

    new_instance.drop('cumulative_size', axis=1, inplace=True)

    # Apply the function to each row of the DataFrame to determine orientation
    new_instance['orientation'] = new_instance.apply(determine_orientation, axis=1)



    # Handle missing or empty values
    new_instance.fillna(-1, inplace=True)  # Assuming missing values are represented as NaN, replace them with 0
    new_instance.replace('', -1, inplace=True)  # Replace empty strings with 0

    # Calculate angles for right elbow, left elbow, right waist, left waist, knee joint, back, armpits/shoulders, and neck
    new_instance['right_elbow_angle'] = new_instance.apply(find_angle,args=('right_shoulder', 'right_elbow', 'right_hand'), axis=1)
    new_instance['left_elbow_angle'] = new_instance.apply(find_angle, args=('left_shoulder', 'left_elbow', 'left_hand'),  axis=1)
    new_instance['right_waist_angle'] = new_instance.apply(find_angle,args=('right_shoulder', 'waist_right', 'right_knee'), axis=1)
    new_instance['left_waist_angle'] = new_instance.apply(find_angle, args=('left_shoulder', 'waist_left', 'left_knee'), axis=1)
    new_instance['right_knee_angle'] = new_instance.apply(find_angle, args=('right_knee', 'waist_right', 'right_foot'),axis=1)
    new_instance['left_knee_angle'] = new_instance.apply(find_angle, args=('left_knee', 'waist_left', 'left_foot'), axis=1)
    new_instance['back_angle'] = new_instance.apply(find_angle, args=('waist_right', 'waist_left', 'left_shoulder'),axis=1)
    new_instance['left_armpits_angle'] = new_instance.apply(find_angle, args=('left_elbow', 'left_shoulder', 'waist_right'), axis=1)
    new_instance['right_armpits_angle'] = new_instance.apply(find_angle,args=('right_elbow', 'right_shoulder', 'waist_left'), axis=1)
    new_instance['neck_angle'] = new_instance.apply(find_angle, args=('right_ear', 'right_shoulder', 'left_shoulder'),axis=1)

    # Calculate Euclidean distance between ear and shoulder
    new_instance['ear_shoulder_distance'] = new_instance.apply(
        lambda row: euclidean_distance(row['right_shoulder_x'], row['right_shoulder_y'], row['right_ear_x'],
                                       row['right_ear_y']), axis=1)
    # Calculate Euclidean distance between hands and elbows
    new_instance['right_hands_elbows_distance'] = new_instance.apply(
        lambda row: euclidean_distance(row['right_hand_x'], row['right_hand_y'], row['right_elbow_x'],
                                       row['right_elbow_y']), axis=1)
    new_instance['left_hands_elbows_distance'] = new_instance.apply(
        lambda row: euclidean_distance(row['left_hand_x'], row['left_hand_y'], row['left_elbow_x'],
                                       row['left_elbow_y']), axis=1)

    # Calculate Euclidean distance between hands and shoulders
    new_instance['right_hands_shoulders_distance'] = new_instance.apply(
        lambda row: euclidean_distance(row['right_hand_x'], row['right_hand_y'], row['right_shoulder_x'],
                                       row['right_shoulder_y']), axis=1)
    new_instance['left_hands_shoulders_distance'] = new_instance.apply(
        lambda row: euclidean_distance(row['left_hand_x'], row['left_hand_y'], row['left_shoulder_x'],
                                       row['left_shoulder_y']), axis=1)



    #
    new_instance['video_name'] = df_copy['video_name']
    # # Convert 'video_name' to integer using LabelEncoder
    #

    # encoder = LabelEncoder()
    # new_instance['video_name'] = encoder.fit_transform(new_instance['video_name'])

    new_instance['video_key'] = pd.factorize(new_instance['video_name'])[0]
    # Replace 'video_key' with 'video_name'
    new_instance['video_name'] = new_instance['video_key']

    # Drop the 'video_key' column
    new_instance.drop('video_key', axis=1, inplace=True)
    with open("label_encoder_orientation.pkl", 'rb') as file:
        label_encoder_orientation.pkl = pickle.load(file)

    with open("label_encoder_orientation.pkl", 'rb') as file:
        label_encoder_orientation.pkl = pickle.load(file)


    return new_instance


def extract_features_from_instance(df):
    # Assume df is your DataFrame with elbow angles
    # df = pd.read_csv('your_csv_file.csv')

    # Define phase thresholds
    curl_up_threshold = 60
    curl_down_threshold = 120

    # Create a new column for phase and initialize with an empty string
    df['bicep_curl_phase'] = ''

    # Iterate through rows and label phases
    for video_name, video_df in df.groupby('video_name'):
        # Sort the DataFrame based on time_interval
        video_df = video_df.sort_values(by='time_interval')

        # Identify the start and end indices for each phase
        curl_up_indices = video_df[video_df['left_elbow_angle'] <= curl_up_threshold].index
        curl_down_indices = video_df[video_df['left_elbow_angle'] >= curl_down_threshold].index

        # Label the phases
        df.loc[curl_up_indices, 'bicep_curl_phase'] = 'Curl Up'
        df.loc[curl_down_indices, 'bicep_curl_phase'] = 'Curl Down'

        encoder = LabelEncoder()
        # Fit and transform the 'video_name' column
        df['video_name'] = encoder.fit_transform(df['video_name'])
        # Convert 'bicep_curl_phase' to integer using LabelEncoder
        # Initialize the LabelEncoder
        phase_encoder = LabelEncoder()

        # Fit and transform the 'bicep_curl_phase' column
        df['bicep_curl_phase'] = phase_encoder.fit_transform(df['bicep_curl_phase'])
        # print(df.dtypes)

    return df



def predict_correctness(df, model):

    # Convert features to a DataFrame
    # features_df = pd.DataFrame([features], columns=features)
    # row_array = df.iloc[0].values

    data_2d_list = df.values.tolist()

    # Use the trained model to make predictions
    prediction = model.predict(data_2d_list)

    with open('encoder.pkl', 'rb') as model_file:
        label_encoder = pickle.load(model_file)

    # Map the numeric prediction back to 'correct' or 'incorrect' using the label encoder
    result = label_encoder.inverse_transform([prediction])[0]

    return result

