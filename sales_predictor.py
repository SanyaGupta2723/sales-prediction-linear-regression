# sales_predictor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Added this import
from sklearn.linear_model import LinearRegression # For Regression Model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # For Regression Evaluation
import matplotlib.pyplot as plt
import seaborn as sns # For plotting

# --- Configuration ---
# IMPORTANT: Set the correct path to your Advertising.csv file
# Using a raw string (r"...") for the absolute path to avoid unicode escape errors.
# Ensure this path exactly matches where you saved Advertising.csv
DATA_PATH = r"C:\Users\Lenovo\Downloads\codesoft\ds task 4\advertising.csv" # Your dataset file name (corrected to lowercase 'a')

# --- Step 1: Load the Dataset ---
print("Loading the Sales Prediction dataset...")
try:
    # Read the CSV file
    df = pd.read_csv(DATA_PATH)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file was not found at {DATA_PATH}")
    print("Please check the DATA_PATH in the script and ensure the CSV file is in the correct folder.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# --- Step 2: Initial Data Inspection ---
print("\n--- Dataset Head (first 5 rows) ---")
print(df.head()) # First 5 rows of the DataFrame

print("\n--- Dataset Info (column types and non-null counts) ---")
df.info() # Information about the DataFrame, including data types and missing values

print("\n--- Dataset Description (statistical summary) ---")
print(df.describe(include='all')) # Statistical summary of all columns (numerical and categorical)

print("\n--- Missing Values Count ---")
print(df.isnull().sum()) # Count of missing values per column

print("\nInitial data loading and inspection complete. Ready for data cleaning and preprocessing.")

# --- Step 3: Data Cleaning and Preprocessing ---

# 3.1 Drop irrelevant columns (e.g., 'Unnamed: 0' if it exists)
# Many CSV files come with an extra index column. Check df.head() to confirm its presence.
print("\n--- Checking for and dropping 'Unnamed: 0' column ---")
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)
    print("Dropped 'Unnamed: 0' column.")
else:
    print("'Unnamed: 0' column not found, no action needed.")

# 3.2 Handle Missing Values (if any)
# Based on typical 'Advertising.csv' dataset, there are usually no missing values.
# But it's good practice to check and handle.
print("\n--- Handling Missing Values (if any) ---")
if df.isnull().sum().sum() > 0: # Check if there are any missing values in the entire DataFrame
    print("Missing values found. Dropping rows with missing values.")
    initial_rows = df.shape[0]
    df.dropna(inplace=True)
    rows_after_dropping = df.shape[0]
    print(f"Dropped {initial_rows - rows_after_dropping} rows with missing values.")
else:
    print("No missing values found. Data is clean.")

print("\n--- Missing Values Count After Cleaning ---")
print(df.isnull().sum()) # Verify no more missing values

# 3.3 Check and Convert Data Types (if necessary)
# For 'Advertising.csv', all columns (TV, Radio, Newspaper, Sales) are usually numeric.
# This step is more for general practice.
print("\n--- Checking Data Types ---")
df.info() # Re-check info after cleaning

print("\nData cleaning and preprocessing complete. Ready for Exploratory Data Analysis (EDA) and model training.")

# --- Step 4: Separate Features (X) and Target (y) ---
# Features are 'TV', 'Radio', 'Newspaper'. Target is 'Sales'.
print("\n--- Separating Features (X) and Target (y) ---")
X = df[['TV', 'Radio', 'Newspaper']] # Features
y = df['Sales'] # Target variable

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# --- Step 5: Split Data into Training and Testing Sets ---
# We split the data to train the model on one part and test its performance on unseen data.
# test_size=0.2 means 20% of data will be used for testing, 80% for training.
# random_state ensures reproducibility of the split.
print("\n--- Splitting Data into Training and Testing Sets ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# --- Step 6: Model Training (Linear Regression) ---
# Linear Regression is a common choice for predicting continuous values.
print("\n--- Training Linear Regression Model ---")
model = LinearRegression()
model.fit(X_train, y_train)
print("Linear Regression Model trained successfully.")

# --- Step 7: Model Evaluation ---
print("\n--- Evaluating Linear Regression Model ---")
y_pred = model.predict(X_test)

# Evaluate using common regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# Optional: Plotting Actual vs. Predicted Sales
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Diagonal line for perfect prediction
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs. Predicted Sales (Linear Regression)')
plt.grid(True)
plt.show()

print("\nSales Prediction Task 4 (Linear Regression) complete.")
