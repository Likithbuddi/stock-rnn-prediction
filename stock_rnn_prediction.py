#!/usr/bin/env python3
"""
Predict next day's closing price using historical stock prices with a SimpleRNN.

Features:
- Downloads historical data from Yahoo Finance (yfinance).
- Creates sequences of past days for training.
- Builds a SimpleRNN model with a Dense regression output.
- Trains the model and evaluates using MAE and RMSE.
- Plots actual vs predicted closing prices.
"""

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
import yfinance as yf


def download_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download historical daily price data from Yahoo Finance."""
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError("No data downloaded. Check ticker symbol and date range.")
    # Keep only the 'Close' column; ensure it's a DataFrame
    df = df[["Close"]].reset_index()
    return df


def create_sequences(data: np.ndarray, look_back: int = 60) -> tuple:
    """
    Reshape an array of closing prices into input/output sequences.

    Parameters
    ----------
    data : np.ndarray
        1-D array of closing prices.
    look_back : int
        Number of previous days to use as input features.

    Returns
    -------
    X, y : tuple of np.ndarray
        X shape: (samples, look_back, 1)
        y shape: (samples,)
    """
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_simple_rnn_model(input_shape: tuple) -> keras.Model:
    """Build a SimpleRNN model with a Dense regression output."""
    model = keras.Sequential(
        [
            # SimpleRNN layer
            layers.SimpleRNN(50, return_sequences=False, input_shape=input_shape),
            # Dense output layer for regression
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def plot_actual_vs_predicted(train_rmse: float, test_rmse: float,
                             train_mae: float, test_mae: float,
                             train_actual, train_pred, test_actual, test_pred,
                             ticker: str) -> None:
    """Plot actual vs predicted closing prices for training and testing sets."""
    plt.figure(figsize=(14, 7))

    # Training plot
    plt.plot(train_actual, label="Train Actual", color="blue", alpha=0.6)
    plt.plot(train_pred, label="Train Predicted", color="red", alpha=0.6)

    # Testing plot
    plt.plot(test_actual, label="Test Actual", color="lightblue", alpha=0.6)
    plt.plot(test_pred, label="Test Predicted", color="orange", alpha=0.6)

    plt.title(f"SimpleRNN - Actual vs Predicted Close Price for {ticker}\n"
              f"Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f} | "
              f"Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f}")
    plt.xlabel("Date (days)")
    plt.ylabel("Closing Price (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="SimpleRNN stock close price predictor")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (e.g., AAPL, GOOG)")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--look-back", type=int, default=60, help="Number of previous days as input")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    args = parser.parse_args()

    # 1. Download data
    try:
        df = download_data(args.ticker, args.start, args.end)
    except Exception as e:
        print(f"Error downloading data: {e}")
        sys.exit(1)

    # 2. Prepare data: scale to [0,1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices = df["Close"].values.reshape(-1, 1)
    scaled = scaler.fit_transform(close_prices).flatten()

    # 3. Create sequences
    X, y = create_sequences(scaled, look_back=args.look_back)
    if len(X) == 0:
        print("Not enough data for the given look-back. Reduce look-back or extend date range.")
        sys.exit(1)

    # 4. Split into train/test (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Reshape input to [samples, time_steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # 5. Build model
    model = build_simple_rnn_model(input_shape=(args.look_back, 1))
    model.summary()

    # 6. Train
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    # 7. Predict
    train_pred_scaled = model.predict(X_train)
    test_pred_scaled = model.predict(X_test)

    # Inverse transform to original price space
    train_pred = scaler.inverse_transform(train_pred_scaled)
    test_pred = scaler.inverse_transform(test_pred_scaled)
    y_train_orig = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # 8. Evaluate
    train_rmse = np.sqrt(np.mean((train_pred.flatten() - y_train_orig) ** 2))
    test_rmse = np.sqrt(np.mean((test_pred.flatten() - y_test_orig) ** 2))
    train_mae = np.mean(np.abs(train_pred.flatten() - y_train_orig))
    test_mae = np.mean(np.abs(test_pred.flatten() - y_test_orig))

    print("\n=== Evaluation ===")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test  RMSE: {test_rmse:.4f}")
    print(f"Train MAE:  {train_mae:.4f}")
    print(f"Test  MAE:  {test_mae:.4f}")

    # 9. Plot actual vs predicted
    plot_actual_vs_predicted(
        train_rmse, test_rmse, train_mae, test_mae,
        y_train_orig, train_pred.flatten(),
        y_test_orig, test_pred.flatten(),
        args.ticker,
    )


if __name__ == "__main__":
    main()