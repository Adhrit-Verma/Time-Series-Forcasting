import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('dataset/AirPassengers.csv')

# Inspect the first few rows
print(df.head())

# Convert the 'Month' column to datetime
df['Month'] = pd.to_datetime(df['Month'])

# Set 'Month' as the index
df.set_index('Month', inplace=True)

# Summary statistics
print(df.describe())

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(df['#Passengers'])  # Correct column name
plt.title('Air Passengers Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.show()
