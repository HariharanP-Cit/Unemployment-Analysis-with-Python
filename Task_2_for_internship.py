import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
# Load the dataset
data = pd.read_csv('Unemployment in India.csv')
# Display the first few rows of the dataset
data.columns = data.columns.str.strip()#added
print(data.head())
# Check for missing values
print(data.isnull().sum())

# Convert 'Date' to datetime

data['Date'] = data['Date'].str.strip()#added


data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')#modified

# Drop rows with missing values
data.dropna(inplace=True)
# Get basic statistics of the dataset
print(data.describe())
# Group by Region and Area to analyze unemployment rates
grouped_data = data.groupby(['Region', 'Area'])['Estimated Unemployment Rate (%)'].mean().reset_index()
print(grouped_data)
# Plotting the unemployment rate over time
plt.figure(figsize=(14, 7))
sns.lineplot(data=data, x='Date', y='Estimated Unemployment Rate (%)', hue='Region')
plt.title('Unemployment Rate Over Time by Region')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.legend(title='Region')
plt.show()
# Filter data for the year 2020
covid_data = data[data['Date'].dt.year == 2020]

# Plotting the unemployment rate during Covid-19
plt.figure(figsize=(14, 7))
sns.lineplot(data=covid_data, x='Date', y='Estimated Unemployment Rate (%)', hue='Region')
plt.title('Impact of Covid-19 on Unemployment Rates (2020)')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.legend(title='Region')
plt.show()
# Set the Date as index
data.set_index('Date', inplace=True)

# Decompose the time series for a specific region
result = seasonal_decompose(data[data['Region'] == 'Andhra Pradesh']['Estimated Unemployment Rate (%)'], model='additive')
result.plot()
plt.show()




