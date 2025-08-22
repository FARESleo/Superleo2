import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from math import isnan

OKX_BASE = "https://www.okx.com"

# ----------------------------
# HTTP helper with retries
# ----------------------------
def okx_get(path, params=None, retries=3, delay=0.6):
    url = f"{OKX_BASE}{path}"
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(delay * (i + 1))
    return None

# ----------------------------
# Data fetchers
# ----------------------------
@st.cache_data(ttl=60)
def fetch_instruments(inst_type="SWAP"):
    j = okx_get("/api/v5/public/instruments", {"instType": inst_type})
    if not j or "data" not in j:
        return []
    return [d["instId"] for d in j["data"]]

@st.cache_data(ttl=45)
def fetch_ohlcv(instId, bar="1H", limit=200):
    j = okx_get("/api/v5/market/candles", {"instId": instId, "bar": bar, "limit": str(limit)})
    if not j or "data" not in j:
        return pd.DataFrame()
    df = pd.DataFrame(j["data"], columns=["ts","o","h","l","c","vol","v2","v3","confirm"])
    for col in ["o","h","l","c","vol"]:
        df[col] = df[col].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
    return df.iloc[::-1].reset_index(drop=True)

@st.cache_data(ttl=20)
def fetch_ticker(instId):
    j = okx_get("/api/v5/market/ticker", {"instId": instId})
    try:
        return float(j["data"][0]["last"])
    except Exception:
        return None

@st.cache_data(ttl=60)
def fetch_funding(instId):
    j = okx_get("/api/v5/public/funding-rate", {"instId": instId})
    try:
        return float(j["data"][0]["fundingRate"])
    except Exception:
        return None

@st.cache_data(ttl=60)
def fetch_oi(instId):
    j = okx_get("/api/v5/public/open-interest", {"instId": instId})
    try:
        return float(j["data"][0]["oi"])
    except Exception:
        return None

# ----------------------------
# Small helpers / metrics
# ----------------------------
def compute_cvd(trades_df):
    if trades_df.empty:
        return None
    signed = np.where(trades_df["side"].str.lower()=="buy", trades_df["sz"], -trades_df["sz"])
    return float(signed.sum())

def orderbook_imbalance(bids, asks):
    if bids is None or asks is None or bids.empty or asks.empty:
        return None
    bid_vol = bids["size"].sum()
    ask_vol = asks["size"].sum()
    imb = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
    return float(imb)

# ----------------------------
# Confidence engine
# ----------------------------
def compute_confidence(instId, bar="1H"):
    ohlcv = fetch_ohlcv(instId, bar, limit=300)
    price = fetch_ticker(instId)
    funding = fetch_funding(instId)
    oi = fetch_oi(instId)

    metrics = {
        "funding": funding if funding else 0.5,
        "oi": oi if oi else 0.5
    }

    # Compute confidence %
    confidence_pct = round((metrics["funding"]*0.1 + metrics["oi"]*0.15 + 0.5*0.75)*100,1)
    if confidence_pct >= 65:
        label = "📈 Bullish"
        recommendation = "Consider LONG (buy)"
    elif confidence_pct <= 35:
        label = "📉 Bearish"
        recommendation = "Consider SHORT (sell)"
    else:
        label = "⚠️ Neutral"
        recommendation = "No clear trend"

    raw = {
        "price": price,
        "funding": funding,
        "oi": oi
    }

    return {
        "label": label,
        "confidence_pct": confidence_pct,
        "recommendation": recommendation,
        "metrics": metrics,
        "raw": raw
    }

# ----------------------------
# Streamlit Dashboard UI
# ----------------------------
st.set_page_config(page_title="Smart Money Scanner V3.6", layout="wide")
st.title("🧠 Smart Money Scanner V3.6 — Color Dashboard")

# Sidebar
inst_type = st.sidebar.selectbox("Instrument Type", ["SWAP","SPOT"])
instruments = fetch_instruments(inst_type)
if not instruments:
    st.sidebar.error("Unable to load instruments from OKX")
    st.stop()

instId = st.sidebar.selectbox("Instrument", instruments)
bar = st.sidebar.selectbox("Timeframe", ["1m","5m","15m","1H","4H","1D"], index=3)
show_raw = st.sidebar.checkbox("Show Raw Metrics")

if st.sidebar.button("Compute Signal"):
    with st.spinner("Fetching live data..."):
        result = compute_confidence(instId, bar)

    # Top signal panel
    st.markdown(f"## {result['label']} — Confidence: {result['confidence_pct']}%")
    st.markdown(f"### Recommendation: {result['recommendation']}")
    st.metric("Live Price", f"{result['raw']['price']:,}" if result['raw']['price'] else "N/A")

    # Colored cards for each metric
    st.markdown("### Metrics Overview")
    cols = st.columns(len(result["metrics"]))
    colors = {"funding":"#f9a825", "oi":"#1e88e5"}
    icons = {"funding":"💰","oi":"📊"}
    idx = 0
    for k,v in result["metrics"].items():
        col = cols[idx]
        val = v
        color = colors.get(k,"#4caf50")
        icon = icons.get(k,"📈")
        col.markdown(f"<div style='background-color:{color};padding:10px;border-radius:10px;text-align:center'><h3>{icon} {k.upper()}</h3><h2>{val:.4f}</h2></div>",unsafe_allow_html=True)
        idx +=1

    if show_raw:
        st.markdown("### Raw Metrics")
        st.json(result["raw"])
else:
    st.info("Select instrument/timeframe and press 'Compute Signal'")
