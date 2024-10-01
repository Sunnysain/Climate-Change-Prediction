import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from main import predict_temperature, predict_precipitation, predict_multivariate
import folium
from streamlit_folium import folium_static  # For rendering Folium maps in Streamlit

# Set the title of the dashboard
st.title("Climate Change Prediction Dashboard - New Delhi: BY DALCHAND SAIN")

# Input date from the user
input_date = st.date_input("Select a date for prediction")

# Sidebar for model selection
model_choice = st.sidebar.selectbox(
    'Choose a Prediction Model',
    ('Temperature Prediction', 'Precipitation Prediction', 'Multivariate Prediction')
)

# When the 'Predict' button is clicked
if st.button('Predict'):
    if model_choice == 'Temperature Prediction':
        # Predict temperature
        temperature = predict_temperature(input_date)
        st.subheader(f"Predicted Average Temperature on {input_date}")
        st.write(f"{temperature:.2f} °C")
        
        # Visualization
        fig, ax = plt.subplots()
        ax.bar(['Predicted Temperature'], [temperature], color='orange')
        st.pyplot(fig)
        
    elif model_choice == 'Precipitation Prediction':
        # Predict precipitation
        precipitation = predict_precipitation(input_date)
        st.subheader(f"Predicted Precipitation on {input_date}")
        st.write(f"{precipitation:.2f} mm")
        
        # Visualization
        fig, ax = plt.subplots()
        ax.bar(['Predicted Precipitation'], [precipitation], color='blue')
        st.pyplot(fig)
        
    else:
        # Predict both temperature and precipitation
        prediction = predict_multivariate(input_date)

        # Ensure the prediction array has the correct shape
        if prediction.size == 2:
            temperature = prediction[0]
            precipitation = prediction[1]
            st.subheader(f"Multivariate Predictions on {input_date}")
            st.write(f"Predicted Average Temperature: {temperature:.2f} °C")
            st.write(f"Predicted Precipitation: {precipitation:.2f} mm")

            # Visualization
            fig, ax1 = plt.subplots()
            ax1.bar(['Temperature'], [temperature], color='orange')
            ax2 = ax1.twinx()
            ax2.bar(['Precipitation'], [precipitation], color='blue', alpha=0.7)
            st.pyplot(fig)

            # Create and display the map
            safdarjung_enclave_coords = (28.563286, 77.191154)
            climate_map = folium.Map(location=safdarjung_enclave_coords, zoom_start=14)

            # Add marker with predictions
            folium.Marker(
                location=safdarjung_enclave_coords,
                popup=(
                    f"Safdarjung Enclave<br>"
                    f"Date: {input_date}<br>"
                    f"Predicted Temp: {temperature:.2f} °C<br>"
                    f"Predicted Precipitation: {precipitation:.2f} mm"
                ),
                icon=folium.Icon(color='blue' if precipitation < 100 else 'red')  # Color based on precipitation
            ).add_to(climate_map)

            # Render the map in Streamlit
            folium_static(climate_map)

        else:
            st.error("Prediction did not return two values. Please check the model.")

# Load historical data for visualization
file_path = r'C:\Climate Change project\3806442.csv'
df = pd.read_csv(file_path)
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')

# Option to display historical data
if st.checkbox('Show Historical Data'):
    st.subheader('Historical Climate Data')
    st.write(df[['DATE', 'TAVG', 'PRCP']].head())

# Option to visualize historical trends
if st.checkbox('Show Historical Trends'):
    st.subheader('Historical Temperature Trends')
    st.line_chart(df.set_index('DATE')['TAVG'])
    
    st.subheader('Historical Precipitation Trends')
    st.line_chart(df.set_index('DATE')['PRCP'])
