
#To Do this vissualization project, you will need the following libraries: Make sure to install them using pip:

#pip install pandas
#pip install matplotlib
#pip install statsmodels
#pip install seaborn
#pip install numpy


#Import the necessary libraries:
import pandas as pd

# Assuming your data is stored in a CSV file
df = pd.read_csv(r'C:\Climate Change project\3806442.csv')


# Data Cleaning and Preprocessing

#Convert the DATE column to a datetime format:
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')


#Handle missing values for columns like TAVG, TMAX, TMIN by filling or interpolating:

# Filling missing values with the mean or interpolation
df['TAVG'] = df['TAVG'].fillna(df['TAVG'].mean())
df['TMIN'] = df['TAVG'].fillna(df['TMIN'].mean())
df['TMAX'] = df['TAVG'].fillna(df['TMAX'].mean())
df['PRCP'] = df['PRCP'].fillna(0)  # Assuming 0 means no precipitation


#Exploratory Data Analysis (EDA)
#Summary statistics: Get a quick overview of the data using:
print(df.describe())

#Visualize temperature trends: Plot the average temperature over time to observe seasonal or annual changes.
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(df['DATE'], df['TAVG'], label='Average Temperature')
plt.title('Average Temperature Over Time (New Delhi)')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

#Precipitation Trends: Visualize how precipitation changes over time.
plt.figure(figsize=(10,6))
plt.bar(df['DATE'], df['PRCP'], color='skyblue', label='Precipitation')
plt.title('Precipitation Over Time (New Delhi)')
plt.xlabel('Date')
plt.ylabel('Precipitation (mm)')
plt.legend()
plt.show()

#Analyze Temperature Extremes

#Maximum and Minimum Temperatures: Analyze how temperature extremes change over time.

plt.figure(figsize=(10,6))
plt.plot(df['DATE'], df['TMAX'], label='Max Temp', color='red')
plt.plot(df['DATE'], df['TMIN'], label='Min Temp', color='blue')
plt.title('Max and Min Temperatures Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

#Seasonal Decomposition
#You can break down the temperature data into trend, seasonality, and residual components using time-series decomposition.
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df['TAVG'], model='additive', period=365)
decomposition.plot()
plt.show()


#Climate Change Analysis
#Long-term trend: Plot a long-term trend line to see if average temperatures are increasing over time.
import seaborn as sns

plt.figure(figsize=(10,6))
sns.regplot(x=df['DATE'].map(pd.Timestamp.toordinal), y=df['TAVG'], scatter=False, label='Trend Line')
plt.title('Long-Term Trend in Average Temperature')
plt.xlabel('Year')
plt.ylabel('Average Temperature (°C)')
plt.legend()
plt.show()

'''The correlation between specific variables like TAVG (average temperature), 
TMIN (minimum temperature), TMAX (maximum temperature), and PRCP (precipitation)'''


# Find correlation between TAVG and PRCP, TMIN and PRCP, TMAX and PRCP
correlation_tavg_prcp = df['TAVG'].corr(df['PRCP'])
correlation_tmin_prcp = df['TMIN'].corr(df['PRCP'])
correlation_tmax_prcp = df['TMAX'].corr(df['PRCP'])

# Print the correlations
print(f"Correlation between TAVG and PRCP: {correlation_tavg_prcp}")
print(f"Correlation between TMIN and PRCP: {correlation_tmin_prcp}")
print(f"Correlation between TMAX and PRCP: {correlation_tmax_prcp}")


'''To visualize the correlation between TAVG, TMIN, TMAX, and PRCP, 
we use a scatter plot with a regression line (to show the linear relationship) for each pair.
The plots visualize:
TAVG (Average Temperature) vs PRCP (Precipitation)
TMIN (Minimum Temperature) vs PRCP (Precipitation)
TMAX (Maximum Temperature) vs PRCP (Precipitation)'''

import seaborn as sns
import matplotlib.pyplot as plt

# Set the style for the plot
sns.set(style="whitegrid")

# Create a 1x3 grid for subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot correlation between TAVG and PRCP
sns.regplot(x='PRCP', y='TAVG', data=df, ax=axes[0], scatter_kws={'s':10}, line_kws={"color": "red"})
axes[0].set_title('TAVG vs PRCP')
axes[0].set_xlabel('PRCP (Precipitation)')
axes[0].set_ylabel('TAVG (Avg Temp)')

# Plot correlation between TMIN and PRCP
sns.regplot(x='PRCP', y='TMIN', data=df, ax=axes[1], scatter_kws={'s':10}, line_kws={"color": "red"})
axes[1].set_title('TMIN vs PRCP')
axes[1].set_xlabel('PRCP (Precipitation)')
axes[1].set_ylabel('TMIN (Min Temp)')

# Plot correlation between TMAX and PRCP
sns.regplot(x='PRCP', y='TMAX', data=df, ax=axes[2], scatter_kws={'s':10}, line_kws={"color": "red"})
axes[2].set_title('TMAX vs PRCP')
axes[2].set_xlabel('PRCP (Precipitation)')
axes[2].set_ylabel('TMAX (Max Temp)')

# Adjust layout to avoid overlapping
plt.tight_layout()

# Show the plot
plt.show()
