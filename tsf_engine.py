import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

import itertools

def eda():    
    # Load dataset
    df = pd.read_csv('dataset/AirPassengers.csv')

    # Inspect the first few rows
    print(df.head())

    # Convert the 'Month' column to datetime
    df['Month'] = pd.to_datetime(df['Month'])

    # Set 'Month' as the index
    df.set_index('Month', inplace=True)

    # Rename column for easier access
    df.rename(columns={'#Passengers': 'Passengers'}, inplace=True)

    # Summary statistics
    print(df.describe())

    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.plot(df['Passengers'])  # Now using renamed column
    plt.title('Air Passengers Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Passengers')
    plt.show()

    # Call ARIMA function for model fitting
    arima_model(df)

def arima_model(df):
    # Check for stationarity using ADF Test
    result = adfuller(df['Passengers'])
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])

    if result[1] > 0.05:
        print("Data is non-stationary, differencing required.")
        # Differencing the data to make it stationary
        df_diff = df['Passengers'].diff().dropna()

        # Re-check stationarity
        result_diff = adfuller(df_diff)
        print('ADF Statistic after differencing:', result_diff[0])
        print('p-value after differencing:', result_diff[1])
    else:
        print("Data is stationary, no differencing needed.")
        df_diff = df['Passengers']

    # Add progress print statement
    print("Starting ARIMA model fitting...")

    # Fitting ARIMA model (p,d,q) - You can tune these parameters
    model = ARIMA(df_diff, order=(5, 1, 0))
    model_fit = model.fit()

    # Summary of the model
    print(model_fit.summary())

    tune_arima_model(df)

def tune_arima_model(df):
    p = range(0, 4)  # Try AR terms from 0 to 3
    d = [1]          # Differencing is fixed at 1
    q = range(0, 4)  # Try MA terms from 0 to 3

    # Create all combinations of p, d, and q
    pdq = list(itertools.product(p, d, q))

    best_aic = float('inf')  # Initialize with a large number
    best_pdq = None
    best_model = None

    # Try different combinations and fit the model
    for param in pdq:
        try:
            model = ARIMA(df['Passengers'], order=param)
            model_fit = model.fit()
            print(f'Tried ARIMA{param} - AIC:{model_fit.aic}')
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_pdq = param
                best_model = model_fit
        except:
            continue

    # Print the best model
    print(f'Best ARIMA{best_pdq} - AIC:{best_aic}')
    print(best_model.summary())


eda()