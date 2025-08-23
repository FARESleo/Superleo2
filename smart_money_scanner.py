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
# Data fetchers (cached)
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
        df["px"] = df["px"].astype(float)
        df["sz"] = df["sz"].astype(float)
        df["side"] = df["side"].astype(str)
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
    except Exception:
        pass
    return df.sort_values("ts").reset_index(drop=True)

# ----------------------------
# Metrics
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
    return float((bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9))

def simple_backtest_winrate(ohlcv_df, lookahead=6, stop_pct=0.01, rr=2.0):
    if ohlcv_df.empty or len(ohlcv_df) < 80:
        return None
    df = ohlcv_df.copy()
    df["ema20"] = df["c"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["c"].ewm(span=50, adjust=False).mean()
    delta = df["c"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100/(1+rs))
    wins=losses=0
    for i in range(50, len(df)-lookahead-1):
        row = df.iloc[i]
        cond = (row["ema20"] > row["ema50"]) and (45 <= row["rsi"] <= 70)
        if not cond:
            continue
        entry = row["c"]
        stop = entry*(1-stop_pct)
        target = entry*(1+stop_pct*rr)
        future = df.iloc[i+1:i+1+lookahead]
        hit_t = future["h"].ge(target).any()
        hit_s = future["l"].le(stop).any()
        if hit_t and not hit_s:
            wins += 1
        else:
            losses += 1
    total = wins + losses
    return (wins/total) if total>0 else None

def compute_support_resistance(ohlcv_df, window=20):
    if ohlcv_df.empty:
        return None, None
    recent = ohlcv_df[-window:]
    support = recent["l"].min()
    resistance = recent["h"].max()
    return support, resistance

def detect_candle_signal(ohlcv_df):
    if ohlcv_df.empty:
        return None
    last = ohlcv_df.iloc[-1]
    prev = ohlcv_df.iloc[-2]
    # bullish engulfing
    if last["c"] > last["o"] and prev["c"] < prev["o"] and last["c"] > prev["o"] and last["o"] < prev["c"]:
        return "Bullish Engulfing"
    # bearish engulfing
    if last["c"] < last["o"] and prev["c"] > prev["o"] and last["c"] < prev["o"] and last["o"] > prev["c"]:
        return "Bearish Engulfing"
    return None

def compute_confidence(instId, bar="1H"):
    ohlcv = fetch_ohlcv(instId, bar, limit=300)
    price = fetch_ticker(instId)
    funding = fetch_funding(instId)
    oi = fetch_oi(instId)
    bids, asks = fetch_orderbook(instId, depth=60)
    trades = fetch_trades(instId, limit=400)

    cvd = compute_cvd(trades) if not trades.empty else None
    ob_imb = orderbook_imbalance(bids, asks)
    bt_win = simple_backtest_winrate(ohlcv, lookahead=6, stop_pct=0.01, rr=2.0)
    support, resistance = compute_support_resistance(ohlcv)
    candle_signal = detect_candle_signal(ohlcv)

    # Normalize metrics 0..1
    fund_score = 0.5 + np.tanh(funding*500)/2 if funding is not None else 0.5
    oi_score = 0.5 + (np.tanh(np.log1p(oi)/20.0))/2 if oi is not None else 0.5
    cvd_score = 0.5 + np.tanh(cvd/1e4)/2 if cvd is not None else 0.5
    ob_score = (ob_imb +1)/2 if ob_imb is not None else 0.5
    bt_score = bt_win if bt_win is not None else 0.5

    metrics = {"funding": fund_score, "oi": oi_score, "cvd": cvd_score, "orderbook": ob_score, "backtest": bt_score}
    weights = {"backtest":0.3, "orderbook":0.25, "cvd":0.2, "oi":0.15, "funding":0.1}
    conf = sum(metrics[k]*weights[k] for k in metrics)
    confidence_pct = round(max(0, min(conf*100,100)),1)

    # Trade suggestion
    if confidence_pct >= 65:
        label = "📈 Bullish"
        recommendation = "LONG (buy)"
        entry = price
        target = round(resistance if resistance else price*1.01,6)
        stop = round(support if support else price*0.99,6)
    elif confidence_pct <= 35:
        label = "📉 Bearish"
        recommendation = "SHORT (sell)"
        entry = price
        target = round(support if support else price*0.99,6)
        stop = round(resistance if resistance else price*1.01,6)
    else:
        label = "⚠️ Neutral / Mixed"
        recommendation = "Wait"
        entry = price
        target = stop = None

    raw = {"price":price,"funding":funding,"oi":oi,"cvd":cvd,"orderbook_imbalance":ob_imb,"backtest_win":bt_win,"support":support,"resistance":resistance,"candle_signal":candle_signal}

    return {"label":label,"confidence_pct":confidence_pct,"recommendation":recommendation,"entry":entry,"target":target,"stop":stop,"metrics":metrics,"weights":weights,"raw":raw}

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Smart Money Scanner V3.8", layout="wide")
st.title("🧠 Smart Money Scanner V3.8 — Flexible Signals & Trade Suggestion")

inst_type = st.sidebar.selectbox("Instrument Type", ["SWAP","SPOT"])
instruments = fetch_instruments(inst_type)
if not instruments:
    st.sidebar.error("Unable to load instruments from OKX.")
    st.stop()
instId = st.sidebar.selectbox("Instrument", instruments, index=0)
bar = st.sidebar.selectbox("Timeframe", ["1m","5m","15m","1H","4H","1D"], index=3)
show_raw = st.sidebar.checkbox("Show Raw metrics", value=False)

if st.sidebar.button("Compute Confidence"):
    with st.spinner("Computing — gathering live data..."):
        result = compute_confidence(instId, bar)

    st.subheader(f"{result['label']} — Confidence: {result['confidence_pct']}%")
    st.markdown(f"### Recommendation: {result['recommendation']}")
    st.metric("Live Price", f"{result['raw']['price']:,}" if result['raw']['price'] else "N/A")

    # Metrics with icons
    icons = {"funding":"💰","oi":"📊","cvd":"📈","orderbook":"⚖️","backtest":"🧪"}
    cols = st.columns(5)
    for idx, k in enumerate(["funding","oi","cvd","orderbook","backtest"]):
        col = cols[idx]
        score = result["metrics"][k]
        weight = result["weights"][k]
        contrib = round(score*weight*100,2)
        col.metric(label=f"{icons[k]} {k.upper()}", value=f"{score:.3f}", delta=f"w={weight}")
        col.caption(f"Contribution: {contrib}%")

    st.markdown("🔎 **Support / Resistance & Candle Signals**")
    st.markdown(f"• Support (approx): {result['raw']['support']}")
    st.markdown(f"• Resistance (approx): {result['raw']['resistance']}")
    st.markdown(f"• Candle Signal: {result['raw']['candle_signal'] if result['raw']['candle_signal'] else 'None'}")

    st.markdown("📝 **Trade Suggestion**")
    st.markdown(f"• Entry: {result['entry']}")
    st.markdown(f"• Target: {result['target'] if result['target'] else 'N/A'}")
    st.markdown(f"• Stop: {result['stop'] if result['stop'] else 'N/A'}")
    st.markdown(f"• Reason: Metrics + Orderbook + Support/Resistance + Candle Signal")

    if show_raw:
        st.markdown("### Raw metrics (for transparency)")
        st.json(result["raw"])
else:
    st.info("Select instrument/timeframe and press 'Compute Confidence' in the sidebar.")
