# temp_model.py

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
file_path = r'C:\Climate Change project\3806442.csv'
df = pd.read_csv(file_path)

# Data preprocessing
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')

# Interpolate missing temperature values
df['TAVG'] = df['TAVG'].interpolate(method='linear')
df['TMAX'] = df['TMAX'].interpolate(method='linear')
df['TMIN'] = df['TMIN'].interpolate(method='linear')

# Feature engineering
df['Year'] = df['DATE'].dt.year
df['Month'] = df['DATE'].dt.month
df['DayOfYear'] = df['DATE'].dt.dayofyear

# Create features (X) and target variable (y)
X = df[['Year', 'Month', 'DayOfYear']]
y = df['TAVG']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply polynomial features
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Train the polynomial regression model
model = LinearRegression()
model.fit(X_poly_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_poly_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (Temperature Model): {mse}')

# Define the directory and model path
model_dir = r'C:\Climate Change project\models'
model_path = os.path.join(model_dir, 'temp_model.pkl')

# Check if the directory exists, create it if not
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Directory created: {model_dir}")

# Save the model and polynomial transformer
with open(model_path, 'wb') as file:
    pickle.dump((model, poly), file)
print(f"Temperature model saved to {model_path}")

# Visualization
plt.figure(figsize=(8, 5))
plt.scatter(X_test['Year'], y_test, color='blue', label='Actual Temperature', alpha=0.6)
plt.scatter(X_test['Year'], y_pred, color='orange', label='Predicted Temperature', alpha=0.6)
plt.title('Actual vs Predicted Average Temperature')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(model_dir, 'temperature_predictions.png'))
plt.show()
