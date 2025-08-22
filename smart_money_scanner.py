import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import time

BASE_URL = "https://www.okx.com"

# =========================
# Utility: Retry requests
# =========================
def okx_get(endpoint, params=None, retries=3, delay=1):
    url = f"{BASE_URL}{endpoint}"
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(delay * (i+1))
    return None

# =========================
# Load Instruments
# =========================
@st.cache_data
def load_instruments(inst_type="SWAP"):
    data = okx_get("/api/v5/public/instruments", {"instType": inst_type})
    if not data or "data" not in data: 
        return []
    return [d["instId"] for d in data["data"]]

# =========================
# Fetch OHLCV
# =========================
def get_ohlcv(instId, bar="1H", limit=200):
    raw = okx_get("/api/v5/market/candles", {"instId": instId, "bar": bar, "limit": str(limit)})
    if not raw or "data" not in raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw["data"], columns=["ts","o","h","l","c","vol","volCcy","volCcyQuote","confirm"])
    df = df.astype({"o":float,"h":float,"l":float,"c":float,"vol":float})
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
    return df.sort_values("ts")

# =========================
# Funding Rate & OI
# =========================
def get_funding(instId):
    raw = okx_get("/api/v5/public/funding-rate", {"instId": instId})
    try:
        return float(raw["data"][0]["fundingRate"])
    except: return None

def get_open_interest(instId):
    raw = okx_get("/api/v5/public/open-interest", {"instId": instId})
    try:
        return float(raw["data"][0]["oi"])
    except: return None

# =========================
# CVD (Cumulative Volume Delta)
# =========================
def get_cvd(instId, limit=200):
    raw = okx_get("/api/v5/market/trades", {"instId": instId, "limit": str(limit)})
    if not raw or "data" not in raw: return pd.DataFrame()
    df = pd.DataFrame(raw["data"], columns=["px","sz","side","ts"])
    df["px"] = df["px"].astype(float)
    df["sz"] = df["sz"].astype(float)
    df["delta"] = np.where(df["side"]=="buy", df["sz"], -df["sz"])
    df["cvd"] = df["delta"].cumsum()
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
    return df.sort_values("ts")

# =========================
# Backtester (1:2 R:R example)
# =========================
def backtest(instId, bar="1H", rr_ratio=2.0):
    df = get_ohlcv(instId, bar=bar, limit=200)
    if df.empty: return None
    wins, losses = 0, 0
    for i in range(len(df)-1):
        entry = df.iloc[i]["c"]
        high = df.iloc[i+1]["h"]
        low = df.iloc[i+1]["l"]
        stop = entry*0.99
        target = entry*(1+(0.01*rr_ratio))
        if high>=target: wins+=1
        elif low<=stop: losses+=1
    total = wins+losses
    return {"wins":wins,"losses":losses,"winrate": wins/total*100 if total>0 else None}

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Smart Money Scanner V3.2", layout="wide")
st.title("📊 Smart Money Scanner – V3.2")

# Sidebar
inst_type = st.sidebar.selectbox("Instrument Type", ["SPOT","SWAP"])
instruments = load_instruments(inst_type)
instId = st.sidebar.selectbox("Select Instrument", instruments)

bar = st.sidebar.selectbox("Timeframe", ["1m","5m","15m","1H","4H","1D"])

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Chart","📉 Derivatives","🔍 CVD","📊 Backtester","⚙ Raw Data"])

# Chart
with tab1:
    df = get_ohlcv(instId, bar)
    if df.empty:
        st.warning("No OHLCV data.")
    else:
        fig = go.Figure(data=[go.Candlestick(x=df["ts"], open=df["o"], high=df["h"], low=df["l"], close=df["c"])])
        st.plotly_chart(fig, use_container_width=True)

# Derivatives
with tab2:
    funding = get_funding(instId)
    oi = get_open_interest(instId)
    st.write(f"**Funding Rate:** {funding}")
    st.write(f"**Open Interest:** {oi}")

# CVD
with tab3:
    df_cvd = get_cvd(instId)
    if df_cvd.empty:
        st.warning("No CVD data.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_cvd["ts"], y=df_cvd["cvd"], mode="lines", name="CVD"))
        st.plotly_chart(fig, use_container_width=True)

# Backtester
with tab4:
    result = backtest(instId, bar)
    if result:
        st.write(result)
    else:
        st.warning("Not enough data for backtest.")

# Raw
with tab5:
    if st.button("Show Raw OHLCV JSON"):
        st.json(okx_get("/api/v5/market/candles", {"instId":instId,"bar":bar,"limit":"5"}))
