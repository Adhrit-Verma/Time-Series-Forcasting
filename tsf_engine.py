import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

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

    # Call LSTM function for model fitting
    lstm_model(df)

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

    # Forecasting with ARIMA
    forecast_arima(df)

def tune_arima_model(df):
    p = range(0, 4)  # Try AR terms from 0 to 3
    d = [1]          # Differencing is fixed at 1
    q = range(0, 4)  # Try MA terms from 0 to 3

    pdq = list(itertools.product(p, d, q))
    best_aic = float('inf')
    best_pdq = None

    for param in pdq:
        try:
            model = ARIMA(df['Passengers'], order=param)
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_pdq = param
        except:
            continue

    print(f'Best ARIMA{best_pdq} - AIC:{best_aic}')
    return best_pdq

def forecast_arima(df):
    best_pdq = tune_arima_model(df)
    model = ARIMA(df['Passengers'], order=best_pdq)
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=12)
    print('ARIMA Forecast:', forecast)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df['Passengers'], label='Actual')
    plt.plot(pd.date_range(df.index[-1], periods=12, freq='MS'), forecast, label='Forecast')
    plt.legend()
    plt.title('ARIMA Forecast')
    plt.show()

def lstm_model(df):
    # Prepare data for LSTM model
    df_scaled, scaler = scale_data(df)
    X_train, y_train = prepare_lstm_data(df_scaled)

    # LSTM Model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Early stopping
    early_stop = EarlyStopping(monitor='loss', patience=5)

    # Fit LSTM model
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, callbacks=[early_stop])

    # Forecasting with LSTM
    forecast_lstm(df, model, scaler)

def forecast_lstm(df, model, scaler):
    last_date = df.index[-1]
    last_values = df['Passengers'].values[-12:].reshape(1, 12, 1)  # Last 12 months

    forecast = []
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='ME')

    for i in range(12):
        pred = model.predict(last_values)
        pred_rescaled = scaler.inverse_transform(pred)  # Inverse scaling
        forecast.append(pred_rescaled[0, 0])  # Append predicted value
        pred_reshaped = pred.reshape(1, 1, 1)
        last_values = np.append(last_values[:, 1:, :], pred_reshaped, axis=1)

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Passengers'], label='Historical Data', color='blue')
    plt.plot(forecast_dates, forecast, label='LSTM Forecast', color='orange')
    plt.legend()
    plt.grid()
    plt.show()

    return forecast, forecast_dates


def scale_data(df):
    # Scaling the data for LSTM (values between 0 and 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['Passengers']])
    return df_scaled, scaler

def prepare_lstm_data(df_scaled, time_step=12):
    # Preparing the data for LSTM
    X, y = [], []
    for i in range(time_step, len(df_scaled)):
        X.append(df_scaled[i-time_step:i, 0])
        y.append(df_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

eda()