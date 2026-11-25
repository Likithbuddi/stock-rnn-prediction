#!/usr/bin/env python3
"""
Train a SimpleRNN on recent AAPL history (2023 -> today).

This script:
  1. Downloads ~2+ years of daily data (2023-01-01 to today).
  2. Runs a small hyper-parameter grid to minimise TEST MSE.
  3. Re-trains the best configuration with full epoch logging.
  4. Saves the best model, scaler and metrics for the Streamlit UI.
"""
import datetime as dt
import itertools
import json
import os
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
TICKER = "AAPL"
START_DATE = "2023-01-01"
END_DATE = (dt.date.today() + dt.timedelta(days=1)).isoformat()  # include today

MODEL_PATH = "simple_rnn_model.keras"
SCALER_PATH = "scaler.save"
METRICS_PATH = "metrics.json"
RESULTS_PATH = "training_results.json"

# Small grid focused on reducing test MSE
GRID = {
    "look_back": [45, 60, 75],
    "units": [128, 192],
    "epochs": [60, 80, 100],
}
BATCH_SIZE = 32
SEED = 42


def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_model(look_back, units):
    keras.backend.clear_session()
    random.seed(SEED)
    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)
    model = keras.Sequential([
        layers.SimpleRNN(units, return_sequences=False, input_shape=(look_back, 1)),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model


def evaluate(model, scaler, X_train, y_train, X_test, y_test):
    train_pred = scaler.inverse_transform(model.predict(X_train, verbose=0)).flatten()
    test_pred = scaler.inverse_transform(model.predict(X_test, verbose=0)).flatten()
    y_train_orig = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    mse_train = float(np.mean((train_pred - y_train_orig) ** 2))
    mse_test = float(np.mean((test_pred - y_test_orig) ** 2))
    rmse_train = float(np.sqrt(mse_train))
    rmse_test = float(np.sqrt(mse_test))
    mae_train = float(np.mean(np.abs(train_pred - y_train_orig)))
    mae_test = float(np.mean(np.abs(test_pred - y_test_orig)))
    return {
        "train_mse": mse_train, "train_rmse": rmse_train, "train_mae": mae_train,
        "test_mse": mse_test, "test_rmse": rmse_test, "test_mae": mae_test,
    }


def main():
    print(f"Downloading {TICKER} data from {START_DATE} to {END_DATE} ...")
    df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
    if df.empty:
        raise RuntimeError("No data downloaded. Check ticker/date range.")
    df = df[["Close"]].reset_index()
    print(f"Downloaded {len(df)} trading days "
          f"({df['Date'].min().date()} -> {df['Date'].max().date()}).\n")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

    # ------------------------------------------------------------------
    # Grid search to minimise test MSE
    # ------------------------------------------------------------------
    results = []
    keys = list(GRID.keys())
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    print(f"=== Grid search over {len(combos)} configurations (test MSE) ===")

    for combo in combos:
        params = dict(zip(keys, combo))
        look_back, units, epochs = params["look_back"], params["units"], params["epochs"]
        X, y = create_sequences(scaled, look_back)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        model = build_model(look_back, units)
        model.fit(X_train, y_train, epochs=epochs, batch_size=BATCH_SIZE,
                  validation_data=(X_test, y_test), verbose=0)

        metrics = evaluate(model, scaler, X_train, y_train, X_test, y_test)
        entry = {"params": params, "metrics": metrics}
        results.append(entry)
        print(f"look_back={look_back:3d} units={units:3d} epochs={epochs:3d} "
              f"-> Test MSE={metrics['test_mse']:8.4f} | "
              f"Test RMSE={metrics['test_rmse']:7.4f} | Test MAE={metrics['test_mae']:7.4f}")

    best = min(results, key=lambda r: r["metrics"]["test_mse"])
    print("\n=== Best configuration ===")
    print(json.dumps(best, indent=2))

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # Final training with best config (full epoch logging)
    # ------------------------------------------------------------------
    look_back = best["params"]["look_back"]
    units = best["params"]["units"]
    epochs = best["params"]["epochs"]

    X, y = create_sequences(scaled, look_back)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    print(f"\n=== Final training: {epochs} epochs, look_back={look_back}, units={units} ===")
    model = build_model(look_back, units)
    model.summary()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1,
    )
    print("=== Final training complete ===")

    metrics = evaluate(model, scaler, X_train, y_train, X_test, y_test)

    # Save an actual-vs-predicted plot for the held-out test set.
    test_pred = scaler.inverse_transform(model.predict(X_test, verbose=0)).flatten()
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(y_test_orig, label="Actual close", color="steelblue", linewidth=1)
    ax.plot(test_pred, label="Predicted close", color="crimson", linewidth=1, alpha=0.85)
    ax.set_title(f"{TICKER} test set: actual vs predicted close")
    ax.set_xlabel("Test sequence index")
    ax.set_ylabel("Close price (USD)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    metrics.update({
        "ticker": TICKER,
        "start_date": START_DATE,
        "end_date": df["Date"].max().date().isoformat(),
        "data_rows": int(len(df)),
        "look_back": look_back,
        "units": units,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "keras_final_loss": float(history.history["loss"][-1]),
        "keras_final_val_loss": float(history.history["val_loss"][-1]),
    })

    print("\n=== Final Metrics (price space, USD) ===")
    for k in ["train_mse", "train_rmse", "train_mae", "test_mse", "test_rmse", "test_mae"]:
        print(f"{k:>12}: {metrics[k]:.4f}")
    print(f"Keras final loss     : {metrics['keras_final_loss']:.6f}")
    print(f"Keras final val_loss : {metrics['keras_final_val_loss']:.6f}")

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved {MODEL_PATH}, {SCALER_PATH}, {METRICS_PATH}, {RESULTS_PATH}, actual_vs_predicted.png")


if __name__ == "__main__":
    main()