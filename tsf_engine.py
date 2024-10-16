import pandas as pd
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

    # Call LSTM function for model fitting
    lstm_model(df)

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

def lstm_model(df):
    # Prepare data for LSTM model
    df_scaled, scaler = scale_data(df)
    X_train, y_train = prepare_lstm_data(df_scaled)

    # LSTM Model with two layers and dropout for regularization
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),  # Adding a second LSTM layer
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')

    # Early stopping to avoid overfitting
    early_stop = EarlyStopping(monitor='loss', patience=5)

    # Fit LSTM model
    print("Training LSTM model...")
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, callbacks=[early_stop])

    # Forecasting with LSTM
    forecast_lstm(df, model, scaler)

def forecast_lstm(df, model, scaler):
    # Last 12 months as input
    last_values = df['Passengers'].values[-12:].reshape(1, 12, 1)

    forecast = []

    print("\n--- LSTM Forecast for Next 12 Months ---")
    for i in range(12):
        # Predict next value
        pred = model.predict(last_values)

        # Clip predictions to avoid extreme values
        pred = np.clip(pred, 0, 1)  # Keep predictions within 0-1 range

        # Inverse scale the prediction back to original scale
        pred_rescaled = scaler.inverse_transform(pred)

        # Append the forecasted value
        forecast.append(pred_rescaled[0, 0])

        # Prepare for the next iteration
        pred_reshaped = pred.reshape(1, 1, 1)
        last_values = np.append(last_values[:, 1:, :], pred_reshaped, axis=1)  # Shift and append new prediction

        print(f"Month {i+1}: {pred_rescaled[0, 0]:.2f} passengers")


def scale_data(df):
    # Scaling the data for LSTM (values between 0 and 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['Passengers']])
    return df_scaled, scaler

def prepare_lstm_data(df_scaled, time_step=12):
    X, y = [], []
    for i in range(time_step, len(df_scaled)):
        X.append(df_scaled[i-time_step:i, 0])
        y.append(df_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

# Run the EDA and modeling
eda()
