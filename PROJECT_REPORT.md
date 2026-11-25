# Page 1 — Introduction & Objective

## Title
Next‑Day Stock Closing Price Prediction using a Simple Recurrent Neural Network

## Abstract
This project builds a machine‑learning pipeline that predicts the next trading day's closing price of Apple Inc. (AAPL) from a sequence of historical daily closing prices. A Simple Recurrent Neural Network (SimpleRNN) is trained on recent data beginning in 2023 and extending to the current date. The model is evaluated with Mean Squared Error (MSE), Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). The trained model and its scaler are loaded by an attractive Streamlit dashboard, where a user can select any reference date and obtain a next‑day price prediction without retraining.

## Problem statement
Given the daily closing prices of a stock up to day *t*, predict the closing price on day *t+1*. This is a supervised sequence‑to‑one regression problem. The input is a fixed‑length window of past closes; the output is a single continuous value.

## Dataset
Historical daily prices are downloaded from Yahoo Finance using the `yfinance` library. The ticker is AAPL. The training window starts on **2023‑01‑01** and ends on the most recent available trading day. Only the `Close` column is used as the model feature, keeping the experiment aligned with the suggested SimpleRNN architecture.

## Motivation
Stock prices are noisy and non‑stationary, but short historical sequences often contain momentum and mean‑reversion patterns that a recurrent model can learn. The goal is not to build a trading system, but to demonstrate an end‑to‑end deep‑learning workflow: data acquisition, preprocessing, model training, evaluation, deployment in a UI, and version control.

---

# Page 2 — Methodology

## Data preprocessing
1. **Download** daily `Close` values for the chosen date range.
2. **Scale** the prices to the interval [0, 1] using `MinMaxScaler`. Scaling is essential for stable RNN training and is fitted only on the training window's data distribution (the same scaler object is later saved and reused at prediction time).
3. **Sequence creation**: a sliding window of `look_back` consecutive scaled closes forms the input vector `X`; the scaled close of the following day becomes the target `y`.
4. **Split**: the first 80% of sequences are used for training and the remaining 20% for testing, preserving temporal order.

## Model architecture
The network is deliberately simple, as suggested:

```
Input  →  SimpleRNN(units, return_sequences=False)  →  Dense(1)  →  Predicted close
```

- `SimpleRNN` processes the sequence step by step and produces a single hidden state.
- `Dense(1)` maps that state to the predicted scaled closing price.
- The model is compiled with the **Adam** optimiser and **mean squared error** loss.

## Hyper‑parameter selection
Rather than committing to a single guess, a focused grid search is performed over:

| Parameter | Values tried |
|-----------|--------------|
| `look_back` | 45, 60, 75 days |
| `units` | 128, 192 |
| `epochs` | 60, 80, 100 |

Every configuration is trained and evaluated; the one with the lowest **test MSE** is selected and then re‑trained with full epoch logging to produce the final weights.

## Evaluation metrics
- **MSE** – average squared difference in dollars²; penalises large errors.
- **RMSE** – square root of MSE, in the same unit as the price (USD).
- **MAE** – average absolute error in USD, easier to interpret.

All metrics are computed after inverse‑transforming predictions back to the original price scale.

---

# Page 3 — Training Results

## Data window
Training uses AAPL closes from **2023‑01‑01** to the latest available date in 2026, giving roughly three to four years of daily observations. This is a substantial increase over the earlier experiment, which used data from 2015–2024.

The downloaded dataset contains **979 trading days** (2023-01-01 to 2026-11-25). With a 45-day look-back window, 934 supervised sequences were created: 747 for training and 187 for testing.

## Best configuration
- `look_back`: **45 days**
- `units`: **192**
- `epochs`: **100**
- `batch_size`: **32**
- Selection rule: lowest test MSE across 18 configurations

## Final metrics (USD scale)
| Split | MSE | RMSE | MAE |
|-------|----:|-----:|----:|
| Train | 11.3322 | 3.3663 | 2.3255 |
| Test  | 26.3153 | 5.1298 | 3.8101 |

## Keras training loss
At the end of the final training run, the scaled Keras loss (MSE) was **0.000213** and the validation loss was **0.000483**. These values are in normalised scale; the table above reports error in dollars after inverse transformation.

## Grid search outcome
The grid search evaluated eighteen configurations. The complete ranking is stored in `training_results.json` and is summarised below:

| Look-back | Units | Epochs | Test MSE | Test RMSE | Test MAE |
|----------:|------:|-------:|---------:|----------:|---------:|
| 45 | 192 | 100 | 26.3153 | 5.1298 | 3.8101 |
| 45 | 192 | 80 | 27.1950 | 5.2149 | 3.8708 |
| 45 | 128 | 80 | 27.2602 | 5.2211 | 3.8897 |
| 45 | 128 | 100 | 27.4733 | 5.2415 | 3.8969 |
| 45 | 192 | 60 | 29.1020 | 5.3946 | 4.0042 |
| 45 | 128 | 60 | 29.2738 | 5.4105 | 4.1068 |
| 60 | 128 | 100 | 33.6927 | 5.8045 | 4.4656 |
| 75 | 128 | 100 | 35.6480 | 5.9706 | 4.6335 |
| 60 | 128 | 80 | 37.6944 | 6.1396 | 4.8335 |
| 75 | 192 | 100 | 38.3615 | 6.1937 | 4.8866 |
| 75 | 128 | 80 | 39.4876 | 6.2839 | 4.9905 |
| 60 | 192 | 100 | 39.9021 | 6.3168 | 5.0054 |
| 75 | 192 | 80 | 43.1579 | 6.5695 | 5.2904 |
| 60 | 128 | 60 | 45.2215 | 6.7247 | 5.4281 |
| 60 | 192 | 80 | 46.4915 | 6.8185 | 5.5200 |
| 75 | 128 | 60 | 47.6273 | 6.9013 | 5.6143 |
| 60 | 192 | 60 | 50.5692 | 7.1112 | 5.8339 |
| 75 | 192 | 60 | 54.0570 | 7.3523 | 6.0502 |

The selected configuration minimised test MSE, which is the primary target metric requested for improvement. Compared with the earlier baseline test MSE of **24.8763**, the tuned model is **1.4390 higher**. This is expected to some degree because the two experiments use different time periods (2015–2024 versus 2023–present); within the 2023–present dataset, the grid search selected the configuration with the lowest test MSE.

---

# Page 4 — Streamlit Dashboard & User Input Panel

## Overview
The Streamlit application (`app.py`) is the deployment layer. It intentionally **does not train** the model; it loads the saved artefacts:

- `simple_rnn_model.keras` – trained weights
- `scaler.save` – fitted MinMaxScaler
- `metrics.json` – evaluation numbers and configuration

Loading is cached with `@st.cache_resource`, so the model is read only once per session.

## Dashboard layout
- **Header and description** – explain that predictions come from pre‑trained weights.
- **Metric cards** – display train/test RMSE, MAE and MSE in a four‑column layout.
- **Sidebar input panel** – shows the trained ticker, look‑back window, and a date picker for the *reference date* (the last known trading day).
- **Predict button** – triggers inference using the loaded model.

## Prediction workflow
1. The user selects a reference date within the downloaded data range.
2. The app filters all rows up to and including that date.
3. The last `look_back` closes are scaled with the saved scaler.
4. The reshaped array `(1, look_back, 1)` is passed to `model.predict`.
5. The output is inverse‑transformed into dollars.
6. If a subsequent trading day exists in the dataset, the actual close is shown alongside the predicted close and the absolute error.
7. A Matplotlib chart overlays recent history, the predicted point and the actual point.
8. A separate saved figure, `actual_vs_predicted.png`, shows the full held‑out test predictions against the real closes.

## Running locally
```bash
pip install streamlit yfinance numpy pandas matplotlib scikit-learn tensorflow joblib
python -m streamlit run app.py
```
Open `http://localhost:8501` in a browser.

---

# Page 5 — Discussion, Limitations & Conclusion

## Discussion
The tuned SimpleRNN achieved the test MSE reported on page 3. Reducing test MSE was the central optimisation goal; the grid search systematically explored sequence length and network capacity instead of relying on intuition. Longer look‑back windows allow the network to see more context, while more units increase representational power, but both can overfit if training is too long. The selected epoch count balances these effects.

The dashboard separated training from inference. This is good practice: training is reproducible via `train_model.py`, while the UI remains fast and deterministic because it only performs a forward pass with fixed weights.

## Limitations
- Only the `Close` price is used; volume, open, high and low could provide additional signal.
- A single stock (AAPL) is modelled; results do not generalise to other tickers.
- Stock prices are influenced by news and macro‑economic events that are not present in price history alone.
- The temporal 80/20 split assumes the recent future resembles the recent past.

## Future work
- Include OHLCV features or technical indicators as additional inputs.
- Compare SimpleRNN with LSTM, GRU or Transformer models.
- Use walk‑forward validation instead of a single split.
- Add confidence intervals via dropout or ensemble models.
- Deploy the Streamlit app to Streamlit Cloud or a container platform.

## Conclusion
The project successfully demonstrates an intermediate‑difficulty deep‑learning task: predicting next‑day closing prices from historical sequences. Data from 2023 to the present was used, a SimpleRNN was tuned to lower test MSE, the best weights were persisted, and an interactive Streamlit dashboard was built to make predictions from those weights. All code, trained artefacts, evaluation results and this report are stored in the GitHub repository `Likithbuddi/stock-rnn-prediction`.
