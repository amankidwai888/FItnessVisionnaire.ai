import pandas as pd
import pickle
import math
# Assuming svm_classifier is your trained SVM classifier
# Define a function to split coordinates and create new columns
def split_coordinates(df, column_name):
    new_columns = ['{}_x'.format(column_name), '{}_y'.format(column_name)]
    df[new_columns] = df[column_name].apply(lambda x: pd.Series(x.split(' ')))
    df.drop(column_name, axis=1, inplace=True)

# Define a function to remove brackets and commas
def clean_coordinates(df, column_name):
    for coordinate_type in ['_x', '_y']:
        df[column_name + coordinate_type] = df[column_name + coordinate_type].str.replace(r'[()]', '')
        df[column_name + coordinate_type] = df[column_name + coordinate_type].str.replace(',', '')
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
    }

    # Apply the same column renaming logic if necessary
    new_instance.rename(columns=column_mapping, inplace=True)

    keypoint_columns = ['right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow',
                        'right_hand', 'left_hand', 'waist_right', 'waist_left',
                        'right_knee', 'left_knee']

    for column in keypoint_columns:
        split_coordinates(new_instance, column)

    for column in keypoint_columns:
        clean_coordinates(new_instance, column)

    df_copy = new_instance["video_name"]
    new_instance = new_instance.drop('video_name', axis=1)

    # Assuming DataFrame 'df'
    new_instance = new_instance.apply(pd.to_numeric, errors='coerce')
    # Calculate angles for right elbow, left elbow, right waist, and left waist
    new_instance['right_elbow_angle'] = new_instance.apply(find_angle, args=('right_shoulder', 'right_elbow', 'right_hand'), axis=1)
    new_instance['left_elbow_angle'] = new_instance.apply(find_angle, args=('left_shoulder', 'left_elbow', 'left_hand'), axis=1)
    new_instance['right_waist_angle'] = new_instance.apply(find_angle, args=('right_shoulder', 'waist_right', 'right_knee'), axis=1)
    new_instance['left_waist_angle'] = new_instance.apply(find_angle, args=('left_shoulder', 'waist_left', 'left_knee'), axis=1)

    new_instance['video_name'] = df_copy

    # Encode categorical variables if needed
    if encoder:
        new_instance['video_name'] = encoder.transform(new_instance['video_name'])

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

    # Drop the rows with an empty 'bicep_curl_phase' (rows not in either phase)
    df = df[df['bicep_curl_phase'] != '']

    return df



def predict_correctness(features, model):
    # Use the trained model to make predictions
    prediction = model.predict([features])[0]

    with open('encoder.pkl', 'rb') as model_file:
        label_encoder = pickle.load(model_file)

    # Map the numeric prediction back to 'correct' or 'incorrect' using the label encoder
    result = label_encoder.inverse_transform([prediction])[0]

    return result

