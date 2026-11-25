import datetime as dt
import json
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib

st.set_page_config(page_title="Stock Predictor | Trained SimpleRNN", layout="wide")

st.title("📈 Next‑Day Close Price Predictor")
st.markdown(
    "This dashboard uses a **pre‑trained SimpleRNN** and its saved scaler "
    "to predict the next trading day's closing price. No retraining happens here – "
    "the best model weights (selected by lowest test MSE) from `simple_rnn_model.keras` "
    "are loaded and reused."
)

# -------------------------------------------------------------------------#
#  Load trained artifacts (only once)
# -------------------------------------------------------------------------#
@st.cache_resource
def load_artifacts():
    model = load_model("simple_rnn_model.keras")
    scaler = joblib.load("scaler.save")
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    return model, scaler, metrics


model, scaler, metrics = load_artifacts()

# Download full historical data (cached) for the trained ticker
@st.cache_data(show_spinner="Fetching historical prices...")
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError("No data downloaded. Check ticker and date range.")
    return df[["Close"]].reset_index()


DATA_START = metrics["start_date"]
# yfinance treats `end` as exclusive, so add one day to include the last trained date.
DATA_END = (dt.date.fromisoformat(metrics["end_date"]) + dt.timedelta(days=1)).isoformat()
df = load_data(metrics["ticker"], DATA_START, DATA_END)

LOOK_BACK = metrics["look_back"]

# -------------------------------------------------------------------------#
#  Sidebar – user input panel
# -------------------------------------------------------------------------#
st.sidebar.header("Prediction Input Panel")
st.sidebar.write(f"**Trained ticker:** `{metrics['ticker']}`")
st.sidebar.write(f"**Look‑back window:** {LOOK_BACK} days")

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

ref_date = st.sidebar.date_input(
    "Reference date (last known trading day)",
    value=max_date,
    min_value=min_date,
    max_value=max_date,
)

predict_btn = st.sidebar.button("Predict Next Day")

# -------------------------------------------------------------------------#
#  Main dashboard – training metrics
# -------------------------------------------------------------------------#
st.subheader(f"Model Performance ({metrics['epochs']}-epoch tuned training)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Train RMSE", f"{metrics['train_rmse']:.4f}")
c2.metric("Test  RMSE", f"{metrics['test_rmse']:.4f}")
c3.metric("Train MAE",  f"{metrics['train_mae']:.4f}")
c4.metric("Test  MAE",  f"{metrics['test_mae']:.4f}")

col5, col6 = st.columns(2)
col5.metric("Train MSE", f"{metrics['train_mse']:.4f}")
col6.metric("Test  MSE", f"{metrics['test_mse']:.4f}")

st.caption(
    f"Data window: {metrics['start_date']} -> {metrics['end_date']} | "
    f"Keras final scaled loss (MSE): {metrics['keras_final_loss']:.6f} | "
    f"Validation loss: {metrics['keras_final_val_loss']:.6f} | "
    f"Epochs: {metrics['epochs']} | Look‑back: {LOOK_BACK}"
)

plot_path = "actual_vs_predicted.png"
if os.path.exists(plot_path):
    st.subheader("Actual vs Predicted (Held‑Out Test Set)")
    st.image(plot_path, use_container_width=True)

# -------------------------------------------------------------------------#
#  Prediction logic using saved weights
# -------------------------------------------------------------------------#
if predict_btn:
    with st.spinner("Making prediction with loaded model weights..."):
        # Find rows up to and including the chosen reference date
        mask = pd.to_datetime(df["Date"]).dt.date <= ref_date
        sub = df.loc[mask]
        if len(sub) < LOOK_BACK:
            st.error(f"Need at least {LOOK_BACK} days of history before {ref_date}. Choose a later date.")
        else:
            window = sub["Close"].values[-LOOK_BACK:].reshape(-1, 1)
            scaled_window = scaler.transform(window)
            X = scaled_window.reshape(1, LOOK_BACK, 1)
            pred_scaled = model.predict(X, verbose=0)
            pred_price = scaler.inverse_transform(pred_scaled)[0][0]

            # Actual next-day close (if available) for comparison
            next_rows = df.loc[pd.to_datetime(df["Date"]).dt.date > ref_date]
            if not next_rows.empty:
                actual_next = float(next_rows.iloc[0]["Close"])
                err = abs(pred_price - actual_next)
                st.success(
                    f"Predicted next close after **{ref_date}**: **${pred_price:.2f}**\n\n"
                    f"Actual next close: **${actual_next:.2f}**  |  Absolute error: **${err:.2f}**"
                )
            else:
                st.success(f"Predicted next close after **{ref_date}**: **${pred_price:.2f}**")

            # Plot recent history + prediction
            fig, ax = plt.subplots(figsize=(10, 5))
            recent = sub.tail(60)
            ax.plot(pd.to_datetime(recent["Date"]), recent["Close"], label="Historical Close", color="steelblue")
            if not next_rows.empty:
                pred_date = pd.to_datetime(next_rows.iloc[0]["Date"])
            else:
                pred_date = pd.Timestamp(ref_date) + pd.Timedelta(days=1)
            ax.scatter([pred_date], [pred_price], color="crimson", s=80,
                       label=f"Predicted ${pred_price:.2f}", zorder=5)
            if not next_rows.empty:
                ax.scatter([pred_date], [actual_next], color="green", s=80,
                           label=f"Actual ${actual_next:.2f}", zorder=5)
            ax.set_title(f"{metrics['ticker']} – Next‑Day Prediction using Saved SimpleRNN Weights")
            ax.set_xlabel("Date")
            ax.set_ylabel("Close Price (USD)")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
else:
    st.info("Select a reference date in the sidebar and click **Predict Next Day**.")

# -------------------------------------------------------------------------#
#  Optional raw data preview
# -------------------------------------------------------------------------#
if st.checkbox("Show historical data preview"):
    st.dataframe(df.tail(15))

st.sidebar.markdown("---")
st.sidebar.caption("Pre‑trained model weights • Streamlit dashboard • Yahoo Finance data")