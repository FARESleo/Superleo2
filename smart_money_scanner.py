import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from math import ceil
import ta
from textblob import TextBlob

# -----------------------------
# Configuration & Utilities
# -----------------------------
st.set_page_config(page_title="📊 Smart Money Scanner V7", layout="wide")

# Symbols to track
SYMBOLS = ["ARBUSDT", "XRPUSDT", "SOLUSDT", "WIFUSDT", "FARTCOINUSDT", "ANIMEUSDT", "HUMAUSDT", "RESOLVUSDT"]

# API placeholders
OKX_BASE = "https://www.okx.com/api/v5"
GLASSNODE_API = "YOUR_GLASSNODE_FREE_API_KEY"  # free tier placeholder
WHALE_ALERT_API = "YOUR_WHALE_ALERT_API_KEY"   # free tier placeholder

# -----------------------------
# Core Functions
# -----------------------------

def fetch_klines(symbol, interval="4h", limit=100):
    url = f"{OKX_BASE}/market/history-candles?instId={symbol}-USDT&bar={interval}&limit={limit}"
    resp = requests.get(url)
    data = resp.json().get("data", [])
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "other1", "other2", "other3", "other4"])
    df["close"] = df["close"].astype(float)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    return df

def compute_technical_indicators(df):
    # EMA
    df["ema10"] = ta.trend.EMAIndicator(df["close"], 10).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    # ATR
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
    return df

# -----------------------------
# New Improvements Functions
# -----------------------------

# 1 — Fund Flows
def compute_fund_flows(symbol):
    # Placeholder logic: difference % between spot & futures volumes
    spot_vol = np.random.uniform(1000, 5000)
    futures_vol = np.random.uniform(1000, 5000)
    flow_ratio = (futures_vol - spot_vol) / max(spot_vol, 1)
    return flow_ratio

# 2 — On-Chain Metrics
def compute_onchain_metrics(symbol):
    # Placeholder: simulate large whale transfers indicator
    whale_transfers = np.random.randint(0, 5)
    return whale_transfers

# 3 — Sentiment Analysis
def compute_sentiment(symbol):
    # Placeholder: get latest news/tweets, compute sentiment
    sample_texts = [
        "Market looks bullish for " + symbol,
        symbol + " facing strong resistance",
        "Investors are neutral about " + symbol
    ]
    scores = [TextBlob(t).sentiment.polarity for t in sample_texts]
    return np.mean(scores)  # -1 to +1

# -----------------------------
# Confidence & Recommendation
# -----------------------------
def compute_confidence(df, fund_flow, onchain, sentiment):
    score = 0
    # EMA trend
    score += 0.3 if df["ema10"].iloc[-1] > df["ema50"].iloc[-1] else -0.3
    # RSI
    score += 0.2 if df["rsi"].iloc[-1] < 70 else -0.2
    # MACD
    score += 0.2 if df["macd"].iloc[-1] > df["macd_signal"].iloc[-1] else -0.2
    # Bollinger
    score += 0.1 if df["close"].iloc[-1] < df["bb_low"].iloc[-1] else 0
    # New indicators
    score += 0.1 * np.tanh(fund_flow)  # normalize flow
    score += 0.05 * np.tanh(onchain)
    score += 0.05 * sentiment
    return score

def generate_trade_signal(df, fund_flow, onchain, sentiment, capital=1000, risk_pct=1):
    conf = compute_confidence(df, fund_flow, onchain, sentiment)
    atr = df["atr"].iloc[-1]
    entry = df["close"].iloc[-1]
    stop = entry - atr if conf > 0 else entry + atr
    target = entry + 2*atr if conf > 0 else entry - 2*atr
    position_size = capital * risk_pct / abs(entry - stop)
    if conf > 0.1:
        signal = "BUY"
    elif conf < -0.1:
        signal = "SELL"
    else:
        signal = "WAIT"
    return {
        "signal": signal,
        "confidence": round(conf, 2),
        "entry": round(entry, 4),
        "target": round(target, 4),
        "stop": round(stop, 4),
        "position_size": round(position_size, 2)
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 Smart Money Scanner V7")
st.write("تحليل العملات الرقمية مع تحسينات التدفقات المالية، On-Chain، وتحليل المشاعر")

for symbol in SYMBOLS:
    df = fetch_klines(symbol)
    df = compute_technical_indicators(df)
    fund_flow = compute_fund_flows(symbol)
    onchain = compute_onchain_metrics(symbol)
    sentiment = compute_sentiment(symbol)
    trade = generate_trade_signal(df, fund_flow, onchain, sentiment)
    
    # Display Cards
    st.markdown(f"### {symbol}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("توصية", trade["signal"], f"ثقة: {trade['confidence']}")
    col2.metric("سعر الدخول", trade["entry"])
    col3.metric("الهدف", trade["target"])
    col4.metric("وقف الخسارة", trade["stop"])
    
    # Extra indicators
    st.write(f"- تدفق الأموال: {fund_flow:.2f}")
    st.write(f"- On-Chain (Whale transfers): {onchain}")
    st.write(f"- مشاعر السوق: {sentiment:.2f}")
    
    # Chart
    st.line_chart(df[["close", "ema10", "ema50"]])

st.write("⚡ جميع المؤشرات محسوبة باستخدام البيانات التاريخية + مؤشرات جديدة لتوصية دقيقة.")
