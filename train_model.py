#!/usr/bin/env python3
"""
Train a SimpleRNN model on Yahoo Finance historical close prices.
Runs for a fixed 50 epochs, prints full training progress, saves the trained
model and the fitted scaler to disk, and reports final MSE / RMSE / MAE.
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
import joblib

# --------------------------------------------------------------------------
# Configuration (change here if needed)
# --------------------------------------------------------------------------
TICKER = "AAPL"
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
LOOK_BACK = 60
EPOCHS = 50
BATCH_SIZE = 32

MODEL_PATH = "simple_rnn_model.keras"
SCALER_PATH = "scaler.save"


def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_model(input_shape):
    model = keras.Sequential([
        layers.SimpleRNN(50, return_sequences=False, input_shape=input_shape),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model


def main():
    print(f"Downloading historical data for {TICKER} ...")
    df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
    if df.empty:
        raise RuntimeError("No data downloaded. Check ticker/date range.")
    df = df[["Close"]].reset_index()
    print(f"Downloaded {len(df)} rows of data.\n")

    # Scale prices
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

    X, y = create_sequences(scaled, LOOK_BACK)
    print(f"Created {len(X)} sequences with look-back={LOOK_BACK}.")

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}\n")

    model = build_model((LOOK_BACK, 1))
    model.summary()

    print(f"\n=== Starting training for {EPOCHS} epochs ===")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1,
    )
    print("=== Training complete ===\n")

    # Final predictions and inverse-transform to dollar scale
    train_pred = scaler.inverse_transform(model.predict(X_train, verbose=0)).flatten()
    test_pred = scaler.inverse_transform(model.predict(X_test, verbose=0)).flatten()
    y_train_orig = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mse_train = np.mean((train_pred - y_train_orig) ** 2)
    rmse_train = np.sqrt(mse_train)
    mae_train = np.mean(np.abs(train_pred - y_train_orig))
    mse_test = np.mean((test_pred - y_test_orig) ** 2)
    rmse_test = np.sqrt(mse_test)
    mae_test = np.mean(np.abs(test_pred - y_test_orig))

    # Also report scaled-space MSE as reported by Keras history
    final_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]

    print("=== Final Metrics (price space, USD) ===")
    print(f"Train MSE  : {mse_train:.4f}")
    print(f"Train RMSE : {rmse_train:.4f}")
    print(f"Train MAE  : {mae_train:.4f}")
    print(f"Test  MSE  : {mse_test:.4f}")
    print(f"Test  RMSE : {rmse_test:.4f}")
    print(f"Test  MAE  : {mae_test:.4f}")
    print(f"\nKeras final loss (scaled MSE)         : {final_loss:.6f}")
    print(f"Keras final val_loss (scaled MSE)     : {final_val_loss:.6f}")

    # Persist model and scaler for the Streamlit UI
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")
    print("You can now run the Streamlit app to make predictions with these weights.")


if __name__ == "__main__":
    main()