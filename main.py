import pandas as pd
import numpy as np
import pickle

def load_model(model_path):
    """
    Load the trained model and polynomial transformer from the given path.
    """
    with open(model_path, 'rb') as f:
        model, poly = pickle.load(f)
    return model, poly

def predict_temperature(date):
    """
    Predict average temperature based on the date using the temperature model.
    """
    model_path = r'C:\Climate Change project\models\temp_model.pkl'
    model, poly = load_model(model_path)
    
    date = pd.Timestamp(date)
    year = date.year
    month = date.month
    day_of_year = date.dayofyear
    
    input_value = np.array([[year, month, day_of_year]])
    input_value_poly = poly.transform(input_value)
    
    prediction = model.predict(input_value_poly)[0]
    return prediction

def predict_precipitation(date):
    """
    Predict precipitation based on the date using the precipitation model.
    """
    model_path = r'C:\Climate Change project\models\precip_model.pkl'
    model, poly = load_model(model_path)
    
    date = pd.Timestamp(date)
    year = date.year
    month = date.month
    day_of_year = date.dayofyear
    
    input_value = np.array([[year, month, day_of_year]])
    input_value_poly = poly.transform(input_value)
    
    prediction = model.predict(input_value_poly)[0]
    return prediction

def predict_multivariate(date):
    """
    Predict average temperature and precipitation based on the date using the multivariate model.
    """
    model_path = r'C:\Climate Change project\models\multivariate_model.pkl'
    model, poly = load_model(model_path)
    
    date = pd.Timestamp(date)
    year = date.year
    month = date.month
    day_of_year = date.dayofyear
    
    input_value = np.array([[year, month, day_of_year]])
    input_value_poly = poly.transform(input_value)
    
    prediction = model.predict(input_value_poly)
    
    if prediction.ndim == 1:
        # If it's already a 1D array, return as is
        return prediction
    else:
        # If it is a 2D array, flatten and return the first row
        return prediction.flatten()

# Example usage
if __name__ == '__main__':
    test_date = '2024-10-01'
    temp_pred = predict_temperature(test_date)
    prcp_pred = predict_precipitation(test_date)
    multivar_pred = predict_multivariate(test_date)

    print(f"Predicted Average Temperature on {test_date}: {temp_pred:.2f} °C")
    print(f"Predicted Precipitation on {test_date}: {prcp_pred:.2f} mm")
    
    if multivar_pred.size == 2:
        print(f"Multivariate Predictions on {test_date}:")
        print(f"  Temperature: {multivar_pred[0]:.2f} °C")
        print(f"  Precipitation: {multivar_pred[1]:.2f} mm")
    else:
        print("Error: Multivariate prediction did not return two values.")
