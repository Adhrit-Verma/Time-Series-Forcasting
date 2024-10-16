import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import itertools

def eda():
    # Load dataset
    df = pd.read_csv('dataset/AirPassengers.csv')

    # Convert the 'Month' column to datetime and set it as index
    df['Month'] = pd.to_datetime(df['Month'], infer_datetime_format=True)
    df = df.set_index('Month')
    df = df.asfreq('MS')  # Set frequency to Month Start

    # Rename column for easier access
    df.rename(columns={'#Passengers': 'Passengers'}, inplace=True)

    # Summary statistics
    print("\n--- Data Summary ---")
    print(df.describe())

    # Call ARIMA function for model fitting
    arima_model(df)


def arima_model(df):
    # Check for stationarity using ADF Test
    result = adfuller(df['Passengers'])
    print("\n--- ADF Test for Stationarity ---")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")

    if result[1] > 0.05:
        print("Data is non-stationary, differencing required.")
        df_diff = df['Passengers'].diff().dropna()
    else:
        print("Data is stationary, no differencing needed.")
        df_diff = df['Passengers']

    # Fitting ARIMA model using best parameters
    forecast_arima(df)

def tune_arima_model(df):
    p = range(0, 4)  # Try AR terms from 0 to 3
    d = [1]          # Differencing is fixed at 1
    q = range(0, 4)  # Try MA terms from 0 to 3

    pdq = list(itertools.product(p, d, q))
    best_aic = float('inf')
    best_pdq = None

    print("\n--- Tuning ARIMA Model ---")
    for param in pdq:
        try:
            model = ARIMA(df['Passengers'], order=param)
            model_fit = model.fit(method_kwargs={"maxiter": 500})
            print(f"ARIMA{param} - AIC:{model_fit.aic:.4f}")
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_pdq = param
        except:
            continue

    print(f"Best ARIMA Model: ARIMA{best_pdq} with AIC: {best_aic:.4f}")
    return best_pdq

def forecast_arima(df):
    best_pdq = tune_arima_model(df)
    model = ARIMA(df['Passengers'], order=best_pdq)
    
    # Fitting the model with additional iterations for better convergence
    model_fit = model.fit(method_kwargs={"maxiter": 500})

    # Forecasting the next 12 months
    forecast = model_fit.forecast(steps=12)
    
    print("\n--- ARIMA Forecast for Next 12 Months ---")
    for i, value in enumerate(forecast, 1):
        print(f"Month {i}: {value:.2f} passengers")

    # Calculate RMSE by comparing forecasted and actual data for the last 12 months in the dataset
    rmse = calculate_rmse(df['Passengers'], model_fit)
    print(f"\n--- RMSE: {rmse:.4f} ---")


def calculate_rmse(actual, model_fit):
    # Using the last 12 months for comparison
    n_obs = 12
    actual_values = actual[-n_obs:]
    predicted_values = model_fit.predict(start=len(actual)-n_obs, end=len(actual)-1)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    return rmse

# Run the EDA and modeling
eda()
