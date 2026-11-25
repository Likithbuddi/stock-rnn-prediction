# Stock RNN Prediction

Next-day AAPL closing-price prediction with a SimpleRNN, trained on data from **2023 to today**, and deployed through an interactive **Streamlit** dashboard that reuses the saved model weights.

## Repository contents

| File | Purpose |
|------|---------|
| `train_model.py` | Downloads data, runs a small grid search, trains the best SimpleRNN, saves model/scaler/metrics. |
| `app.py` | Streamlit dashboard with user input panel; loads saved weights and predicts next-day close. |
| `stock_rnn_prediction.py` | Original standalone training script (2015–2024 baseline). |
| `simple_rnn_model.keras` | Best trained model weights. |
| `scaler.save` | Fitted `MinMaxScaler`. |
| `metrics.json` | Final evaluation metrics and configuration. |
| `training_results.json` | Full grid-search results. |
| `actual_vs_predicted.png` | Actual vs predicted close prices on the held-out test set. |
| `PROJECT_REPORT.md` | Five-page project report. |
| `requirements.txt` | Python dependencies. |

## Quick start

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501`, choose a reference date in the sidebar, and click **Predict Next Day**.

## Retrain

```bash
python train_model.py
```

This overwrites `simple_rnn_model.keras`, `scaler.save`, `metrics.json`, `training_results.json` and `actual_vs_predicted.png`.

## Model

```
Input(look_back) -> SimpleRNN(units) -> Dense(1)
```

Optimiser: Adam. Loss: MSE. Metrics: MSE, RMSE, MAE (reported in USD after inverse scaling).
