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
def clean_coordinates(df, column_name):
    for coordinate_type in ['_x', '_y']:
        # Check if the new columns exist before using .str accessor
        if column_name + coordinate_type in df.columns and pd.api.types.is_string_dtype(
                df[column_name + coordinate_type]):
            df[column_name + coordinate_type] = df[column_name + coordinate_type].str.replace(r'[()]', '')
            df[column_name + coordinate_type] = df[column_name + coordinate_type].str.replace(',', '')
        else:
            # Handle non-string data type or missing columns
            pass


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


    # Handle missing or empty values
    new_instance.fillna(0, inplace=True)  # Assuming missing values are represented as NaN, replace them with 0
    new_instance.replace('', 0, inplace=True)  # Replace empty strings with 0

    df_copy = pd.DataFrame(columns=["video_name"])
    df_copy['video_name'] = new_instance["video_name"]

    new_instance = new_instance.apply(pd.to_numeric, errors='coerce')
    # Calculate angles for right elbow, left elbow, right waist, and left waist
    new_instance['right_elbow_angle'] = new_instance.apply(find_angle, args=('right_shoulder', 'right_elbow', 'right_hand'), axis=1)
    new_instance['left_elbow_angle'] = new_instance.apply(find_angle, args=('left_shoulder', 'left_elbow', 'left_hand'), axis=1)
    new_instance['right_waist_angle'] = new_instance.apply(find_angle, args=('right_shoulder', 'waist_right', 'right_knee'), axis=1)
    new_instance['left_waist_angle'] = new_instance.apply(find_angle, args=('left_shoulder', 'waist_left', 'left_knee'), axis=1)

    #
    new_instance['video_name'] = df_copy['video_name']
    # # Convert 'video_name' to integer using LabelEncoder
    #
    # encoder = LabelEncoder()
    # new_instance['video_name'] = encoder.fit_transform(new_instance['video_name'])

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

