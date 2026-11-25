import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
import math

st.set_page_config(page_title="Stock Price Predictor - SimpleRNN", layout="wide")

st.title("📈 SimpleRNN Stock Close‑Price Predictor")
st.markdown(
    "Predict the next day's closing price using a simple Recurrent Neural Network "
    "trained on historical Yahoo Finance data. Adjust the parameters in the sidebar "
    "and see the model's performance in real time."
)

# -------------------------------------------------------------------------#
#  Sidebar user‑input panel
# -------------------------------------------------------------------------#
st.sidebar.header("Input Parameters")

ticker = st.sidebar.text_input("Stock ticker", value="AAPL").upper()
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime("2024-12-31"))
look_back = st.sidebar.slider("Look‑back days (sequence length)", 10, 120, 60)
epochs = st.sidebar.slider("Training epochs", 1, 50, 20)
batch_size = st.sidebar.slider("Batch size", 16, 256, 32)

# -------------------------------------------------------------------------#
#  Helper functions (mirrored from the standalone script)
# -------------------------------------------------------------------------#
@st.cache_data(show_spinner="Downloading data...")
def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError("No data downloaded. Check ticker symbol and date range.")
    df = df[["Close"]].reset_index()
    return df


def create_sequences(data: np.ndarray, look_back: int):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_model(input_shape):
    model = keras.Sequential(
        [
            layers.SimpleRNN(50, return_sequences=False, input_shape=input_shape),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# -------------------------------------------------------------------------#
#  Main workflow
# -------------------------------------------------------------------------#
if st.sidebar.button("Run Prediction"):
    try:
        with st.spinner("Downloading data..."):
            df = download_data(ticker, str(start_date), str(end_date))

        # Scale
        scaler = MinMaxScaler(feature_range=(0, 1))
        close_vals = df["Close"].values.reshape(-1, 1)
        scaled = scaler.fit_transform(close_vals).flatten()

        # Sequences
        X, y = create_sequences(scaled, look_back)
        if len(X) == 0:
            st.error("Not enough data for the chosen look‑back. Reduce look‑back or extend the date range.")
            st.stop()

        # Train / test split (80/20)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Build & train
        model = build_model(input_shape=(look_back, 1))

        with st.spinner("Training model..."):
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                verbose=0,
            )

        # Predict
        train_pred_scaled = model.predict(X_train, verbose=0)
        test_pred_scaled = model.predict(X_test, verbose=0)

        train_pred = scaler.inverse_transform(train_pred_scaled).flatten()
        test_pred = scaler.inverse_transform(test_pred_scaled).flatten()
        y_train_orig = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Metrics
        train_rmse = np.sqrt(np.mean((train_pred - y_train_orig) ** 2))
        test_rmse = np.sqrt(np.mean((test_pred - y_test_orig) ** 2))
        train_mae = np.mean(np.abs(train_pred - y_train_orig))
        test_mae = np.mean(np.abs(test_pred - y_test_orig))

        # ------------------------------------------------------------------
        # Display results
        # ------------------------------------------------------------------
        st.success("Prediction completed!")

        # Metric cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Train RMSE", f"{train_rmse:.2f}")
        col2.metric("Test  RMSE", f"{test_rmse:.2f}")
        col3.metric("Train MAE",  f"{train_mae:.2f}")
        col4.metric("Test  MAE",  f"{test_mae:.2f}")

        # Actual vs Predicted plot
        st.subheader("Actual vs Predicted Closing Price")
        fig, ax = plt.subplots(figsize=(10, 5))

        # Align x‑axis with original dates for visual reference
        train_dates = df["Date"].values[look_back : look_back + len(train_pred)]
        test_dates = df["Date"].values[look_back + split_idx : look_back + split_idx + len(test_pred)]

        ax.plot(train_dates, y_train_orig, label="Train Actual", color="steelblue", alpha=0.6)
        ax.plot(train_dates, train_pred, label="Train Predicted", color="orange", alpha=0.6)
        ax.plot(test_dates, y_test_orig, label="Test Actual", color="lightblue", alpha=0.6)
        ax.plot(test_dates, test_pred, label="Test Predicted", color="salmon", alpha=0.6)

        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price (USD)")
        ax.set_title(f"{ticker} – Actual vs Predicted (RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f})")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        # Optionally show the underlying data table
        if st.checkbox("Show raw data preview"):
            st.dataframe(df.tail(10))

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please check the ticker symbol, date range, and try again.")
else:
    st.info("Adjust the sidebar parameters and click **Run Prediction** to start.")


# -------------------------------------------------------------------------#
#  Footer
# -------------------------------------------------------------------------#
st.sidebar.markdown("---")
st.sidebar.caption(
    "Built with Streamlit • TensorFlow/Keras • Yahoo Finance data • 2026"
)