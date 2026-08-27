# ============================================================
# Kronos vs Prophet — head-to-head backtest on AVAX/USD
# Run this in Google Colab (colab.research.google.com) — not
# part of the automated GitHub Actions alert workflow.
# ============================================================

# !pip install -q requests pandas numpy torch huggingface_hub einops safetensors prophet plotly
# !git clone -q https://github.com/shiyu-coder/Kronos.git
import sys, os
sys.path.insert(0, "/content/Kronos")

import requests, pandas as pd, numpy as np
from model import Kronos, KronosTokenizer, KronosPredictor
from prophet import Prophet
import plotly.graph_objects as go

# --- 1. Fetch AVAX/USD candles ---
def fetch_kraken_klines(pair="AVAXUSD", interval=240, limit=1000):
    url = "https://api.kraken.com/0/public/OHLC"
    resp = requests.get(url, params={"pair": pair, "interval": interval}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(data["error"])
    result_key = [k for k in data["result"].keys() if k != "last"][0]
    raw = data["result"][result_key]
    cols = ["time", "open", "high", "low", "close", "vwap", "volume", "count"]
    df = pd.DataFrame(raw, columns=cols)
    df["timestamps"] = pd.to_datetime(df["time"], unit="s")
    for c in ["open", "high", "low", "close", "volume", "vwap"]:
        df[c] = df[c].astype(float)
    df["amount"] = df["volume"] * df["vwap"]
    return df.tail(limit).reset_index(drop=True)[
        ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    ]

full_df = fetch_kraken_klines("AVAXUSD", 240, 1000)
print(f"Pulled {len(full_df)} candles, {full_df['timestamps'].iloc[0]} to {full_df['timestamps'].iloc[-1]}")

# --- 2. Hold out the last 48 candles (8 days) as "the future" we'll test against ---
HOLDOUT = 48
train_df = full_df.iloc[:-HOLDOUT].reset_index(drop=True)
actual_future = full_df.iloc[-HOLDOUT:].reset_index(drop=True)

print(f"Training on {len(train_df)} candles, testing against {HOLDOUT} held-out actual candles")
print(f"Actual price at cutoff: ${train_df['close'].iloc[-1]:.3f}")
print(f"Actual price 8 days later: ${actual_future['close'].iloc[-1]:.3f}  "
      f"({(actual_future['close'].iloc[-1]/train_df['close'].iloc[-1]-1)*100:+.2f}%)")

# --- 3. Kronos forecast (blind to the holdout) ---
print("\nLoading Kronos-small...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

hist = train_df.tail(400).reset_index(drop=True)
x_df = hist[["open", "high", "low", "close", "volume", "amount"]]
x_timestamp = hist["timestamps"]
step = x_timestamp.diff().median()
last_ts = x_timestamp.iloc[-1]
y_timestamp = pd.Series([last_ts + step * (i + 1) for i in range(HOLDOUT)])

print("Running Kronos forecast...")
kronos_forecast = predictor.predict(
    df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
    pred_len=HOLDOUT, T=1.0, top_p=0.9, sample_count=5,
)
kronos_forecast.index = y_timestamp

# --- 4. Prophet forecast (same training window, same holdout horizon) ---
print("\nRunning Prophet forecast...")
prophet_df = train_df[["timestamps", "close"]].rename(columns={"timestamps": "ds", "close": "y"})
prophet_model = Prophet(daily_seasonality=True, weekly_seasonality=True)
prophet_model.fit(prophet_df)

future = prophet_model.make_future_dataframe(periods=HOLDOUT, freq="4h")
prophet_result = prophet_model.predict(future)
prophet_forecast = prophet_result.tail(HOLDOUT).set_index("ds")["yhat"]

# --- 5. Compare both against actual ---
actual_close = actual_future.set_index("timestamps")["close"]

def rmse(pred, actual):
    return np.sqrt(np.mean((pred.values - actual.values) ** 2))

def mape(pred, actual):
    return np.mean(np.abs((pred.values - actual.values) / actual.values)) * 100

kronos_rmse = rmse(kronos_forecast["close"], actual_close)
kronos_mape = mape(kronos_forecast["close"], actual_close)
prophet_rmse = rmse(prophet_forecast, actual_close)
prophet_mape = mape(prophet_forecast, actual_close)

print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"Kronos  — RMSE: ${kronos_rmse:.3f}  MAPE: {kronos_mape:.2f}%")
print(f"Prophet — RMSE: ${prophet_rmse:.3f}  MAPE: {prophet_mape:.2f}%")
winner = "Kronos" if kronos_rmse < prophet_rmse else "Prophet"
print(f"\nLower error wins: {winner}")

# --- 6. Visual comparison ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=actual_close.index, y=actual_close.values,
                          mode="lines", name="Actual", line=dict(color="black", width=3)))
fig.add_trace(go.Scatter(x=kronos_forecast.index, y=kronos_forecast["close"],
                          mode="lines", name=f"Kronos (RMSE ${kronos_rmse:.2f})",
                          line=dict(color="#2563eb", dash="dash")))
fig.add_trace(go.Scatter(x=prophet_forecast.index, y=prophet_forecast.values,
                          mode="lines", name=f"Prophet (RMSE ${prophet_rmse:.2f})",
                          line=dict(color="#dc2626", dash="dash")))
fig.update_layout(title="Kronos vs Prophet — 8-Day AVAX/USD Forecast Backtest",
                   yaxis_title="AVAX/USD", template="plotly_white", height=550)
fig.show()
