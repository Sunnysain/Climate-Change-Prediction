import pandas as pd
import folium
import webbrowser

# Assuming you've implemented the prediction logic here or in another file.
from main import predict_temperature, predict_precipitation

def create_map(date):
    # Get predicted values for the specified date
    predicted_temperature = predict_temperature(date)
    predicted_precipitation = predict_precipitation(date)

    # Location coordinates for Safdarjung Enclave
    safdarjung_enclave_coords = (28.563286, 77.191154)

    # Create a Folium map centered around Safdarjung Enclave
    map_delhi = folium.Map(location=safdarjung_enclave_coords, zoom_start=14)

    # Add marker with predictions
    folium.Marker(
        location=safdarjung_enclave_coords,
        popup=(
            f"Safdarjung Enclave<br>"
            f"Date: {date}<br>"
            f"Predicted Temp: {predicted_temperature:.2f} °C<br>"
            f"Predicted Precipitation: {predicted_precipitation:.2f} mm"
        ),
        icon=folium.Icon(color='blue' if predicted_precipitation < 100 else 'red')  # Color based on precipitation
    ).add_to(map_delhi)

    # Save the map to an HTML file
    map_path = 'safdarjung_enclave_climate_map.html'
    map_delhi.save(map_path)

    # Automatically open the HTML file in the default web browser
    webbrowser.open(map_path)

    print(f"Map has been saved and opened in your default browser.")

# Example usage
if __name__ == '__main__':
    create_map('2024-10-01')  # You can change this date for predictions
