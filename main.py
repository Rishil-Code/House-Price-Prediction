import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------- 📂 Load Dataset --------------------
print("\U0001f4c2 📂 Loading dataset...\n")

df = pd.read_csv(r"G:\APP Projects\House Price Prediction\data\house_cleaned.csv")

# Display dataset info
print("\✅ Dataset Loaded Successfully!")
print(df.info())
print("\n📌 Columns in dataset:", df.columns)

# -------------------- 🏢 Data Preprocessing --------------------
# Fill missing numerical values with the median
df.fillna(df.select_dtypes(include=['number']).median(), inplace=True)

# Convert categorical features into numerical format using one-hot encoding
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
categorical_encoded = one_hot_encoder.fit_transform(df[categorical_columns])

# Convert to DataFrame with proper column names
categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=one_hot_encoder.get_feature_names_out(categorical_columns))

# Reset index to match original df
categorical_encoded_df.index = df.index

# Drop original categorical columns and concatenate encoded features
df.drop(columns=categorical_columns, inplace=True)
df = pd.concat([df, categorical_encoded_df], axis=1)

# Drop unnecessary columns if they exist
drop_columns = ['rating', 'features']
df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)

# Define features and target variable
X = df.drop(columns=['price'])  # Assuming 'price' is the target
y = df['price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\u2705 Data Preprocessing Complete!")

# -------------------- 🏠 Train Model --------------------
print("🚀 Training Model...\n")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("\u2705 Model Trained Successfully!")

# -------------------- 📊 Model Evaluation --------------------
# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"🔹 MAE: {mae}")
print(f"🔹 MSE: {mse}")
print(f"🔹 RMSE: {rmse}")
print(f"🔹 R-Squared: {r2}")

# -------------------- 💾 Save Model & Scaler --------------------
joblib.dump(model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(one_hot_encoder, 'one_hot_encoder.pkl')

print("\n💾 Model & Scaler Saved!")

# -------------------- 🚀 Predict New House Price --------------------
def predict_price(sample_input):
    # Convert the input to a DataFrame
    sample_input_df = pd.DataFrame([sample_input])

    # Load OneHotEncoder
    one_hot_encoder = joblib.load('one_hot_encoder.pkl')

    # Get categorical and numerical columns from training
    categorical_columns = [col for col in X.columns if col in one_hot_encoder.feature_names_in_]
    numerical_columns = [col for col in X.columns if col not in categorical_columns]

    # Ensure all categorical columns exist in the input
    for col in categorical_columns:
        if col not in sample_input_df.columns:
            sample_input_df[col] = "unknown"

    # Select and encode categorical features
    sample_input_categorical = sample_input_df[categorical_columns]
    sample_input_categorical_encoded = one_hot_encoder.transform(sample_input_categorical)
    sample_input_categorical_df = pd.DataFrame(sample_input_categorical_encoded, 
                                               columns=one_hot_encoder.get_feature_names_out())

    # Ensure numerical columns exist in the input
    for col in numerical_columns:
        if col not in sample_input_df.columns:
            sample_input_df[col] = 0

    # Select numerical features
    sample_input_numerical = sample_input_df[numerical_columns]
    
    # Combine numerical and categorical features
    sample_input_final = pd.concat([sample_input_numerical, sample_input_categorical_df], axis=1)
    
    # Align with training feature order
    sample_input_final = sample_input_final.reindex(columns=X.columns, fill_value=0)
    
    # Standardize
    scaler = joblib.load('scaler.pkl')
    sample_input_scaled = scaler.transform(sample_input_final)
    
    # Predict
    model = joblib.load('house_price_model.pkl')
    predicted_price = model.predict(sample_input_scaled)[0]
    print("\n 🏡 Predicted House Price:", predicted_price)
    return predicted_price

# -------------------- 🏡 Dynamic House Price Prediction --------------------
def get_user_input():
    sample_input = {}

    # Get numerical inputs
    sample_input["price_per_sqft"] = float(input("Enter price per sqft: "))
    sample_input["area"] = float(input("Enter area (sqft): "))
    sample_input["bedRoom"] = int(input("Enter number of bedrooms: "))
    sample_input["bathroom"] = int(input("Enter number of bathrooms: "))
    sample_input["floorNum"] = int(input("Enter floor number: "))

    # Get categorical inputs
    sample_input["property_type"] = input("Enter property type (e.g., Apartment, Villa): ").strip()
    sample_input["society"] = input("Enter society name: ").strip()

    return sample_input

# Get user input dynamically
user_input = get_user_input()
predict_price(user_input)