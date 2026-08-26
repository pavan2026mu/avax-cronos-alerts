import safetensors
import requests, pandas as pd, numpy as np, sys, subprocess, os

if not os.path.exists("Kronos"):
    subprocess.run(["git", "clone", "-q", "https://github.com/shiyu-coder/Kronos.git"])
sys.path.insert(0, "Kronos")
from model import Kronos, KronosTokenizer, KronosPredictor

NTFY_TOPIC = "Avax-cronos-PawanM"
LOCAL_TZ = "America/Los_Angeles"  # PST/PDT, auto-adjusts for daylight saving


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
    df["timestamps"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(LOCAL_TZ)
    for c in ["open", "high", "low", "close", "volume", "vwap"]:
        df[c] = df[c].astype(float)
    df["amount"] = df["volume"] * df["vwap"]
    return df.tail(limit).reset_index(drop=True)[
        ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    ]


def add_indicators(df):
    df = df.copy()
    df["bb_mid"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std

    ema12 = df["volume"].ewm(span=12, adjust=False).mean()
    ema26 = df["volume"].ewm(span=26, adjust=False).mean()
    macd_vol = ema12 - ema26
    df["macd_vol_hist"] = macd_vol - macd_vol.ewm(span=9, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    rmin, rmax = rsi.rolling(14).min(), rsi.rolling(14).max()
    stoch_rsi = (rsi - rmin) / (rmax - rmin)
    df["stoch_k"] = (stoch_rsi * 100).rolling(3).mean()
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    return df


df = add_indicators(fetch_kraken_klines("AVAXUSD", 240, 1000))

print("Loading Kronos-small model + tokenizer...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

lookback, pred_len = 400, 48
hist = df.tail(lookback).reset_index(drop=True)
x_df = hist[["open", "high", "low", "close", "volume", "amount"]]

# predictor needs naive (tz-less) timestamps internally
x_timestamp = hist["timestamps"].dt.tz_localize(None)
step = x_timestamp.diff().median()
last_ts = x_timestamp.iloc[-1]
y_timestamp = pd.Series([last_ts + step * (i + 1) for i in range(pred_len)])

print("Running forecast (may take a minute on CPU)...")
forecast = predictor.predict(
    df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
    pred_len=pred_len, T=1.0, top_p=0.9, sample_count=5,
)
forecast.index = y_timestamp

current = df["close"].iloc[-1]
current_time = df["timestamps"].iloc[-1]
pct_move = (forecast["close"].iloc[-1] / current - 1) * 100

# --- Short notification (unchanged) ---
msg = (
    f"AVAX ${current:.3f} @ {current_time.strftime('%b %d %I:%M %p %Z')}\n"
    f"Kronos 8d forecast: {pct_move:+.2f}%\n"
    f"BB: {df['bb_lower'].iloc[-1]:.2f}-{df['bb_upper'].iloc[-1]:.2f} "
    f"(mid {df['bb_mid'].iloc[-1]:.2f})\n"
    f"StochRSI K={df['stoch_k'].iloc[-1]:.0f} D={df['stoch_d'].iloc[-1]:.0f}\n"
    f"MACD(vol): {df['macd_vol_hist'].iloc[-1]:.0f}"
)

resp = requests.post(
    f"https://ntfy.sh/{NTFY_TOPIC}",
    data=msg.encode("utf-8"),
    headers={"Title": "AVAX Kronos Update"},
)
print(f"ntfy response: {resp.status_code} {resp.text}")
print(msg)

# --- Save full 48-candle detailed forecast to history CSV ---
os.makedirs("forecast_history", exist_ok=True)

detailed = forecast[["open", "high", "low", "close"]].copy()
detailed.insert(0, "forecast_timestamp", detailed.index)
detailed.insert(0, "run_time", current_time)
detailed.insert(0, "run_id", current_time.strftime("%Y%m%d_%H%M%S"))
detailed["current_price_at_run"] = current

history_file = "forecast_history/kronos_forecasts.csv"
detailed.to_csv(
    history_file,
    mode="a",
    header=not os.path.exists(history_file),
    index=False,
)
print(f"Appended {len(detailed)} forecast rows to {history_file}")
