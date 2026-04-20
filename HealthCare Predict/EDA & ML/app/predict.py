import pandas as pd
from joblib import load
import os

# Load the trained machine learning models for the two demographic segments
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_young = load(os.path.join(BASE_DIR, 'artifacts', 'model_young.joblib'))
model_rest = load(os.path.join(BASE_DIR, 'artifacts', 'model_rest.joblib'))

# Load the corresponding scalers used during model training to ensure consistent feature scaling
scaler_young = load(os.path.join(BASE_DIR, 'artifacts', 'scaler_young.joblib'))
scaler_rest = load(os.path.join(BASE_DIR, 'artifacts', 'scaler_rest.joblib'))



# 🔥 Load UPDATED model if exists
young_updated_path = os.path.join(BASE_DIR, 'artifacts', 'model_young_updated.joblib')
rest_updated_path = os.path.join(BASE_DIR, 'artifacts', 'model_rest_updated.joblib')

if os.path.exists(young_updated_path):
    model_young = load(young_updated_path)
else:
    model_young = load(os.path.join(BASE_DIR, 'artifacts', 'model_young.joblib'))

if os.path.exists(rest_updated_path):
    model_rest = load(rest_updated_path)
else:
    model_rest = load(os.path.join(BASE_DIR, 'artifacts', 'model_rest.joblib'))
    

def calculate_normalization_risk(medical_history):
    """
    Calculates a normalized risk score based on the user's medical history.
    This replicates the feature engineering step performed during model training.
    """
    # Define baseline risk scores for various conditions
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }

    # Split the input string to handle multiple conditions (e.g., "Diabetes & Thyroid")
    diseases = medical_history.lower().split(' & ')
    # Calculate the total score by summing the individual disease scores
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)
    
    # Normalize the score to be between 0 and 1, assuming a max possible score of 14 (Diabetes + Heart disease)
    min_score, max_score = 0, 14
    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)
    return normalized_risk_score

def handle_scaling(age, df):
    """
    Applies the appropriate MinMax scaling to the input data based on the user's age.
    """
    # Select the correct scaler dictionary based on age segmentation
    if age <= 25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    # Extract the scaler instance and the list of columns it expects
    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    # 'income_level' was part of the original scaling process but dropped later for the 'young' model.
    # We add it temporarily as a dummy value so the scaler doesn't throw a shape error.
    df['income_level'] = 0
    # Apply the scaling transformation
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    
    # Note: 'income_level' should ideally be dropped here if the models were trained without it,
    # but keeping it ensures the DataFrame matches the expected input shape if the model requires it.
    df.drop('income_level', axis=1, inplace=True)
    return df

def preprocess_input(input_dict):
    """
    Takes the raw input dictionary from the UI and converts it into a structured pandas DataFrame
    ready for prediction. This includes mapping numerical values, encoding categorical variables,
    and calculating derived features.
    """
    # Define the exact columns the ML models expect, in the exact order
    expected_columns = [
        'age', 'number_of_dependants', 'income_level', 'income_lakhs', 'insurance_plan', 'genetical_risk',
        'normalized_risk_score','gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest',
        'marital_status_Unmarried','bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight',
        'smoking_status_Occasional','smoking_status_Regular', 'employment_status_Salaried',
        'employment_status_Self-Employed'
    ]

    # Map direct numerical inputs from the UI keys to the DataFrame column names
    direct_map = {
        'Age': 'age',
        'Number of Dependants': 'number_of_dependants',
        'Income in Lakhs': 'income_lakhs',
        'Genetical Risk': 'genetical_risk',
    }

    # Map categorical choices to their corresponding one-hot encoded columns
    # If a user selects 'Male' for 'Gender', the 'gender_Male' column gets a 1.
    # Note: Base categories (like 'Female' or 'Normal BMI') are represented by 0s in all related columns (drop_first=True in training).
    categorical_map = {
        ('Gender', 'Male'):                      'gender_Male',
        ('Region', 'Northwest'):                 'region_Northwest',
        ('Region', 'Southeast'):                 'region_Southeast',
        ('Region', 'Southwest'):                 'region_Southwest',
        ('Marital Status', 'Unmarried'):         'marital_status_Unmarried',
        ('BMI Category', 'Obesity'):             'bmi_category_Obesity',
        ('BMI Category', 'Overweight'):          'bmi_category_Overweight',
        ('BMI Category', 'Underweight'):         'bmi_category_Underweight',
        ('Smoking Status', 'Occasional'):        'smoking_status_Occasional',
        ('Smoking Status', 'Regular'):           'smoking_status_Regular',
        ('Employment Status', 'Salaried'):       'employment_status_Salaried',
        ('Employment Status', 'Self-Employed'):  'employment_status_Self-Employed',
    }

    # Ordinal encoding mapping for insurance plans
    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    # Initialize an empty DataFrame filled with zeros with the expected columns
    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    # Iterate through the user inputs and populate the DataFrame
    for key, value in input_dict.items():
        if key in direct_map:
            # Handle direct numerical mappings
            df[direct_map[key]] = value
        elif key == 'Insurance Plan':
            # Handle ordinal encoding for Insurance Plan
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif (key, value) in categorical_map:
            # Handle one-hot encoding: set the specific target column to 1
            df[categorical_map[(key, value)]] = 1

    # Calculate and assign the engineered risk score based on medical history text
    df['normalized_risk_score'] = calculate_normalization_risk(input_dict['Medical History'])
    
    # Scale the continuous features appropriately based on age segment
    df = handle_scaling(input_dict['Age'], df)
    return df


def predict(input_dict):
    """
    Main entry point for generating a prediction.
    It preprocesses the input data and routes it to the appropriate model based on age.
    """
    # Format raw UI input into the DataFrame format needed by the models
    input_df = preprocess_input(input_dict)

    # Route prediction to the correct segmented model
    if input_dict['Age'] <= 25:
        # Use the model trained specifically on young demographic data (which includes genetical_risk)
        prediction = model_young.predict(input_df)
    else:
        # Use the model trained on the older demographic
        prediction = model_rest.predict(input_df)
        
    # Return the predicted premium amount as an integer
    return int(prediction[0])
