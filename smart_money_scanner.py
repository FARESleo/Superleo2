
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import datetime

# ================== Helpers ==================
OKX_BASE = "https://www.okx.com/api/v5"

def get_ohlcv(symbol="BTC-USDT-SWAP", bar="1H", limit=100):
    url = f"{OKX_BASE}/market/candles?instId={symbol}&bar={bar}&limit={limit}"
    data = requests.get(url).json()
    if "data" not in data: return pd.DataFrame()
    df = pd.DataFrame(data["data"], columns=[
        "ts","o","h","l","c","vol","volCcy","volCcyQuote","confirm"
    ])
    df = df.astype({"o":float,"h":float,"l":float,"c":float,"vol":float})
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.iloc[::-1].reset_index(drop=True)

def get_funding(symbol="BTC-USDT-SWAP"):
    url = f"{OKX_BASE}/public/funding-rate?instId={symbol}"
    data = requests.get(url).json()
    try:
        return float(data["data"][0]["fundingRate"])
    except:
        return None

def get_open_interest(symbol="BTC-USDT-SWAP"):
    url = f"{OKX_BASE}/public/open-interest?instId={symbol}"
    data = requests.get(url).json()
    try:
        return float(data["data"][0]["oi"])
    except:
        return None

def get_orderbook(symbol="BTC-USDT-SWAP", depth=20):
    url = f"{OKX_BASE}/market/books?instId={symbol}&sz={depth}"
    data = requests.get(url).json()
    if "data" not in data: return pd.DataFrame(), pd.DataFrame()
    ob = data["data"][0]
    bids = pd.DataFrame(ob["bids"], columns=["price","size","liq"]).astype(float)
    asks = pd.DataFrame(ob["asks"], columns=["price","size","liq"]).astype(float)
    return bids, asks

# ================== Indicators ==================
def ema(series, period=20):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta>0,0)).rolling(period).mean()
    loss = (-delta.where(delta<0,0)).rolling(period).mean()
    rs = gain/loss
    return 100 - (100/(1+rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# ================== Backtester ==================
def backtest(df, rr_ratio=2, stop_perc=0.01):
    wins, losses = 0, 0
    for i in range(len(df)-1):
        entry = df["c"].iloc[i]
        stop = entry * (1-stop_perc)
        target = entry * (1+stop_perc*rr_ratio)
        future = df["c"].iloc[i+1:i+5]  # next 5 candles lookahead
        if any(f >= target for f in future):
            wins += 1
        elif any(f <= stop for f in future):
            losses += 1
    total = wins+losses
    return {"winrate": wins/total*100 if total>0 else 0,
            "wins": wins, "losses": losses, "trades": total}

# ================== UI ==================
st.set_page_config(page_title="Smart Money Scanner V2", layout="wide")
st.title("📊 Smart Money Scanner V2 (OKX)")

symbol = st.text_input("Instrument (OKX format)", "BTC-USDT-SWAP")
bar = st.selectbox("Timeframe", ["1m","5m","15m","1H","4H","1D"])

tabs = st.tabs(["Chart","Derivatives","Liquidity Map","Backtester"])

# ---- Chart ----
with tabs[0]:
    df = get_ohlcv(symbol, bar)
    if df.empty:
        st.error("No data available")
    else:
        df["EMA20"] = ema(df["c"], 20)
        df["RSI"] = rsi(df["c"])
        macd_line, signal_line, hist = macd(df["c"])
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df["ts"], open=df["o"], high=df["h"], low=df["l"], close=df["c"],
            name="Price"))
        fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA20"], line=dict(color="orange"), name="EMA20"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("RSI & MACD")
        st.line_chart(df[["RSI"]])

# ---- Derivatives ----
with tabs[1]:
    funding = get_funding(symbol)
    oi = get_open_interest(symbol)
    col1, col2 = st.columns(2)
    col1.metric("Funding Rate", f"{funding:.4%}" if funding else "N/A")
    col2.metric("Open Interest", f"{oi:,.0f}" if oi else "N/A")

# ---- Liquidity Map ----
with tabs[2]:
    bids, asks = get_orderbook(symbol)
    if bids.empty or asks.empty:
        st.error("No orderbook data")
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bids["price"], y=bids["size"], name="Bids", marker_color="green"))
        fig.add_trace(go.Bar(x=asks["price"], y=asks["size"], name="Asks", marker_color="red"))
        st.plotly_chart(fig, use_container_width=True)

# ---- Backtester ----
with tabs[3]:
    if not df.empty:
        rr = st.selectbox("Risk:Reward", [2,3])
        stop = st.slider("Stop %", 0.5, 5.0, 1.0)/100
        result = backtest(df, rr, stop)
        st.write(f"Winrate: {result['winrate']:.2f}%")
        st.write(f"Trades: {result['trades']} | Wins: {result['wins']} | Losses: {result['losses']}")
    else:
        st.info("Load chart first")
