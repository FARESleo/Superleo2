import requests
import pandas as pd
import time
import streamlit as st
from datetime import datetime

OKX_BASE = "https://www.okx.com"

# HTTP helper with retries
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

# Data fetchers (cached)
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
    df = pd.DataFrame(j["data"])
    rename_map = {}
    if "px" not in df.columns and "price" in df.columns:
        rename_map["price"] = "px"
    if "sz" not in df.columns and "size" in df.columns:
        rename_map["size"] = "sz"
    if rename_map:
        df = df.rename(columns=rename_map)
    try:
        df["px"] = pd.to_numeric(df["px"], errors='coerce')
        df["sz"] = pd.to_numeric(df["sz"], errors='coerce')
        df["ts"] = pd.to_numeric(df["ts"], errors='coerce')
        df.dropna(subset=["px", "sz", "ts"], inplace=True)
        df["px"] = df["px"].astype(float)
        df["sz"] = df["sz"].astype(float)
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True, errors='coerce')
        df.dropna(subset=["ts"], inplace=True)
        df["side"] = df["side"].astype(str)
    except Exception:
        return pd.DataFrame()
    return df.sort_values("ts").reset_index(drop=True)

@st.cache_data(ttl=60)
def get_live_market_data():
    try:
        all_coins = []
        for page in range(1, 3):
            url = f"https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 200,
                "page": page,
                "price_change_percentage": "24h"
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                all_coins.extend(response.json())
            else:
                st.error(f"Failed to fetch data from CoinGecko. Status code: {response.status_code}")
                return pd.DataFrame()
        return pd.DataFrame(all_coins)
    except Exception as e:
        st.error(f"Error fetching live data: {e}")
        return pd.DataFrame()
