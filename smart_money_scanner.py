import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from math import isnan
from datetime import datetime

# --- كود إضافة صورة الخلفية ---
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://i.imgur.com/Utvjk6E.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

OKX_BASE = "https://www.okx.com"

# ----------------------------
# HTTP helper with retries
# ----------------------------
@st.cache_data(ttl=600)
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
# Data fetchers (cached)
# ----------------------------
@st.cache_data(ttl=600)
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

@st.cache_data(ttl=30)
def fetch_orderbook(instId, depth=40):
    j = okx_get("/api/v5/market/books", {"instId": instId, "sz": str(depth)})
    if not j or "data" not in j:
        return None, None
    ob = j["data"][0]
    def to_df(raw, side):
        try:
            df = pd.DataFrame(raw)
            if df.shape[1] == 2:
                df.columns = ["price","size"]
            elif df.shape[1] >= 3:
                df = df.iloc[:,:3]
                df.columns = ["price","size","liq"]
            else:
                return pd.DataFrame()
            df = df.astype(float)
            df["side"] = side
            return df
        except Exception:
            return pd.DataFrame()
    bids = to_df(ob.get("bids", []), "bid")
    asks = to_df(ob.get("asks", []), "ask")
    return bids, asks

@st.cache_data(ttl=30)
def fetch_trades(instId, limit=400):
    j = okx_get("/api/v5/market/trades", {"instId": instId, "limit": str(limit)})
    if not j or "data" not in j:
        return pd.DataFrame()
