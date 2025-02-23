import joblib
import numpy as np
import pandas as pd

print("Debug: Script is running...")
print("Starting prediction script...")

# Load the trained model, scaler, and OneHotEncoder
model = joblib.load(r"G:\APP Projects\House Price Prediction\models\house_price_model.pkl")
scaler = joblib.load(r"G:\APP Projects\House Price Prediction\models\scaler.pkl")
one_hot_encoder = joblib.load(r"G:\APP Projects\House Price Prediction\models\one_hot_encoder.pkl")

print("Model, scaler, and encoder loaded successfully!")

# Load training data to get column structure
df_train = pd.read_csv(r"G:\APP Projects\House Price Prediction\data\house_cleaned.csv")
train_columns = df_train.drop(columns=['price']).columns  # Drop target column

# Define a sample input (Ensure these match actual training feature names)
sample_data = {
    'area': [5000],
    'bedRoom': [3],
    'bathroom': [2],
    'balcony': [1],
    'price_per_sqft': [1200],
    'property_type': ['Apartment'],
    'facing': ['North-East']
}

# Convert to DataFrame
df_input = pd.DataFrame(sample_data)

# Separate categorical and numerical features
categorical_cols = ['property_type', 'facing']
numerical_cols = ['area', 'bedRoom', 'bathroom', 'balcony', 'price_per_sqft']

# Encode categorical features using the OneHotEncoder
df_input_categorical = df_input[categorical_cols]

# Transform categorical features
df_input_categorical_encoded = one_hot_encoder.transform(df_input_categorical)

# Convert the result to a DataFrame
df_input_categorical_df = pd.DataFrame(df_input_categorical_encoded.toarray(),
                                       columns=one_hot_encoder.get_feature_names_out(categorical_cols))

# Merge numerical and categorical data
df_input_final = pd.concat([df_input[numerical_cols], df_input_categorical_df], axis=1)

# Ensure all training columns exist in input and add missing ones
for col in train_columns:
    if col not in df_input_final.columns:
        df_input_final[col] = 0  # Add missing columns with default values

# Ensure correct column order
df_input_final = df_input_final[train_columns]

# Scale input data
df_input_scaled = scaler.transform(df_input_final)

print(f"Processed input shape: {df_input_scaled.shape}")

# Make prediction
try:
    prediction = model.predict(df_input_scaled)
    print(f"Predicted Price: {prediction[0]}")
except Exception as e:
    print(f"Error during prediction: {e}")
