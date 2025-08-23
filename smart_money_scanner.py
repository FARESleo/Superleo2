import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from math import isnan
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

OKX_BASE = "https://www.okx.com"
LOCAL_TZ = ZoneInfo("Africa/Algiers")

# ==============================
# Utils: formatting
# ==============================
def format_price(x):
    if x is None:
        return "N/A"
    try:
        x = float(x)
        if x >= 1000:
            return f"{x:,.2f}"
        elif x >= 1:
            return f"{x:,.3f}"
        else:
            return f"{x:.6f}".rstrip("0").rstrip(".") if "." in f"{x:.6f}" else f"{x:.6f}"
    except:
        return str(x)

def format_num(x, decimals=3):
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.{decimals}f}"
    except:
        return str(x)

def clamp01(v):
    try:
        return max(0.0, min(1.0, float(v)))
    except:
        return 0.5

def utc_now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def local_now_str():
    return datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")

# ==============================
# HTTP helper with retries
# ==============================
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

# ==============================
# Data fetchers (cached)
# ==============================
@st.cache_data(ttl=600)
def fetch_instruments(inst_type="SWAP"):
    j = okx_get("/api/v5/public/instruments", {"instType": inst_type})
    if not j or "data" not in j:
        return []
    return [d["instId"] for d in j["data"]]

@st.cache_data(ttl=45)
def fetch_ohlcv(instId, bar="1H", limit=300):
    j = okx_get("/api/v5/market/candles", {"instId": instId, "bar": bar, "limit": str(limit)})
    if not j or "data" not in j:
        return pd.DataFrame()
    df = pd.DataFrame(j["data"], columns=["ts","o","h","l","c","vol","v2","v3","confirm"])
    for col in ["o","h","l","c","vol"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ts"] = pd.to_datetime(pd.to_numeric(df["ts"], errors="coerce"), unit="ms", utc=True)
    df = df.dropna(subset=["o","h","l","c","ts"]).iloc[::-1].reset_index(drop=True)
    return df

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

# Robust orderbook: handle rows of length 2 or 3, avoid ValueError
@st.cache_data(ttl=30)
def fetch_orderbook(instId, depth=60):
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
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["size"] = pd.to_numeric(df["size"], errors="coerce")
            if "liq" in df.columns:
                df["liq"] = pd.to_numeric(df["liq"], errors="coerce")
            df = df.dropna(subset=["price","size"])
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
    # normalize columns
    if "px" not in df.columns and "price" in df.columns:
        df = df.rename(columns={"price": "px"})
    if "sz" not in df.columns and "size" in df.columns:
        df = df.rename(columns={"size": "sz"})
    try:
        df["px"] = pd.to_numeric(df["px"], errors='coerce')
        df["sz"] = pd.to_numeric(df["sz"], errors='coerce')
        df["ts"] = pd.to_numeric(df["ts"], errors='coerce')
        df.dropna(subset=["px","sz","ts"], inplace=True)
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True, errors='coerce')
        df.dropna(subset=["ts"], inplace=True)
        if "side" in df.columns:
            df["side"] = df["side"].astype(str)
        else:
            df["side"] = "buy"
    except Exception:
        return pd.DataFrame()
    return df.sort_values("ts").reset_index(drop=True)

# ==============================
# Metrics & signals
# ==============================
def compute_cvd(trades_df):
    if trades_df is None or trades_df.empty:
        return None
    signed = np.where(trades_df["side"].str.lower()=="buy", trades_df["sz"], -trades_df["sz"])
    return float(np.nansum(signed))

def orderbook_imbalance(bids, asks):
    if bids is None or asks is None or bids.empty or asks.empty:
        return None
    bid_vol = bids["size"].sum()
    ask_vol = asks["size"].sum()
    return float((bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9))

def find_liquidity_zones(bids_df, asks_df, price_range_pct=0.01, top_n=3):
    if bids_df is None or asks_df is None or bids_df.empty or asks_df.empty:
        return [], []
    mid = (bids_df["price"].iloc[0] + asks_df["price"].iloc[0]) / 2.0
    bid_z = bids_df[bids_df["price"] > mid * (1 - price_range_pct)]
    ask_z = asks_df[asks_df["price"] < mid * (1 + price_range_pct)]
    tb = bid_z.nlargest(top_n, "size")[["price","size"]].to_dict("records")
    ta = ask_z.nlargest(top_n, "size")[["price","size"]].to_dict("records")
    return tb, ta

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
    recent = ohlcv_df.tail(window)
    return float(recent["l"].min()), float(recent["h"].max())

def detect_candle_signal(ohlcv_df, bar):
    if ohlcv_df.empty or len(ohlcv_df) < 3:
        return None
    last = ohlcv_df.iloc[-1]
    prev = ohlcv_df.iloc[-2]

    # Engulfing
    if last["c"] > last["o"] and prev["c"] < prev["o"] and (last["c"] >= prev["o"]) and (last["o"] <= prev["c"]):
        return "Bullish Engulfing"
    if last["c"] < last["o"] and prev["c"] > prev["o"] and (last["c"] <= prev["o"]) and (last["o"] >= prev["c"]):
        return "Bearish Engulfing"

    # Morning Star (approx on lower TFs)
    if bar in ["5m","15m","1H"]:
        if len(ohlcv_df) >= 3:
            prev2 = ohlcv_df.iloc[-3]
            is_bear2 = prev2["c"] < prev2["o"]
            is_small_prev = abs(prev["o"] - prev["c"]) < ((prev["h"] - prev["l"]) * 0.2)
            is_bull_last = last["c"] > last["o"]
            if is_bear2 and is_small_prev and is_bull_last:
                return "Bullish Morning Star"
    return None

def calculate_atr(ohlcv_df, period=14):
    if ohlcv_df.empty or len(ohlcv_df) < period + 1:
        return None
    df = ohlcv_df.copy()
    high = df['h']
    low = df['l']
    close = df['c']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])

# ==============================
# Confidence + trade suggestion
# ==============================
def compute_confidence(instId, bar="1H"):
    # fetch
    ohlcv = fetch_ohlcv(instId, bar, limit=300)
    price = fetch_ticker(instId)
    funding = fetch_funding(instId)
    oi = fetch_oi(instId)
    bids, asks = fetch_orderbook(instId, depth=60)
    trades = fetch_trades(instId, limit=400)

    # derived
    cvd = compute_cvd(trades)
    ob_imb = orderbook_imbalance(bids, asks)
    top_bids, top_asks = find_liquidity_zones(bids, asks, top_n=3)
    bt_win = simple_backtest_winrate(ohlcv, lookahead=6, stop_pct=0.01, rr=2.0)
    support, resistance = compute_support_resistance(ohlcv)
    candle_signal = detect_candle_signal(ohlcv, bar)
    atr = calculate_atr(ohlcv, period=14)

    # Normalize to 0..1
    fund_score = 0.5 + np.tanh((funding or 0.0) * 500.0)/2.0 if funding is not None else 0.5
    oi_score   = 0.5 + np.tanh(np.log1p(max(0.0,(oi or 0.0))) / 20.0)/2.0 if oi is not None else 0.5
    cvd_score  = 0.5 + np.tanh((cvd or 0.0) / 1e4)/2.0 if cvd is not None else 0.5
    ob_score   = ((ob_imb or 0.0) + 1.0)/2.0 if ob_imb is not None else 0.5
    bt_score   = bt_win if bt_win is not None else 0.5

    metrics = {
        "funding": clamp01(fund_score),
        "oi": clamp01(oi_score),
        "cvd": clamp01(cvd_score),
        "orderbook": clamp01(ob_score),
        "backtest": clamp01(bt_score),
    }

    weights = {"backtest":0.30, "orderbook":0.25, "cvd":0.20, "oi":0.15, "funding":0.10}
    conf = sum(metrics[k]*weights[k] for k in metrics.keys())
    confidence_pct = round(max(0.0, min(conf*100.0, 100.0)), 1)

    # classify
    if confidence_pct >= 65:
        bias = "bull"
    elif confidence_pct <= 35:
        bias = "bear"
    else:
        bias = "neutral"

    # Signal strength rules
    is_bullish_strong = (
        bias == "bull" and
        (cvd is not None and cvd > 0) and
        (ob_imb is not None and ob_imb > 0) and
        (candle_signal in ["Bullish Engulfing","Bullish Morning Star"])
    )
    is_bullish_weak = (
        (confidence_pct >= 50) and
        ((cvd is not None and cvd > 0) or (candle_signal in ["Bullish Engulfing","Bullish Morning Star"]))
    )
    is_bearish_strong = (
        bias == "bear" and
        (cvd is not None and cvd < 0) and
        (ob_imb is not None and ob_imb < 0) and
        (candle_signal == "Bearish Engulfing")
    )
    is_bearish_weak = (
        (confidence_pct <= 50) and
        ((cvd is not None and cvd < 0) or (candle_signal == "Bearish Engulfing"))
    )

    # Trade plan
    if atr is None or price is None:
        label = "⚠️ Neutral / Mixed"
        recommendation = "Wait"
        strength = "N/A"
        entry = price
        target = stop = None
        reason = "بيانات غير كافية (ATR/Price)."
        color = "gray"
        icon = "⏸️"
    else:
        if is_bullish_strong:
            label, recommendation, strength, color, icon = "📈 Bullish", "LONG", "Strong", "green", "🚀"
            entry = price
            target = entry + (atr * 2.0)
            stop   = entry - (atr * 1.0)
            reason = f"Confluence قوي: {candle_signal} + CVD إيجابي + ميزان العُمق لصالح المشترين."
        elif is_bullish_weak:
            label, recommendation, strength, color, icon = "📈 Bullish", "LONG", "Weak", "green", "🟢"
            entry = price
            target = entry + (atr * 1.5)
            stop   = entry - (atr * 1.0)
            reason = f"ميل صعودي: {candle_signal or 'لا توجد شمعة قوية'} أو CVD إيجابي، لكن الإشارات مختلطة."
        elif is_bearish_strong:
            label, recommendation, strength, color, icon = "📉 Bearish", "SHORT", "Strong", "red", "🔻"
            entry = price
            target = entry - (atr * 2.0)
            stop   = entry + (atr * 1.0)
            reason = f"Confluence قوي: {candle_signal} + CVD سلبي + ميزان العُمق لصالح البائعين."
        elif is_bearish_weak:
            label, recommendation, strength, color, icon = "📉 Bearish", "SHORT", "Weak", "red", "🟥"
            entry = price
            target = entry - (atr * 1.5)
            stop   = entry + (atr * 1.0)
            reason = f"ميل هبوطي: {candle_signal or 'لا توجد شمعة قوية'} أو CVD سلبي، لكن الإشارات مختلطة."
        else:
            label, recommendation, strength, color, icon = "⚠️ Neutral / Mixed", "Wait", "Neutral", "gray", "⏸️"
            entry = price
            target = stop = None
            reason = "لا يوجد سبب مقنع للدخول الآن. انتظر مزيداً من التوافق بين الإشارات."

    raw = {
        "price": price,
        "funding": funding,
        "oi": oi,
        "cvd": cvd,
        "orderbook_imbalance": ob_imb,
        "backtest_win": bt_win,
        "support": support,
        "resistance": resistance,
        "candle_signal": candle_signal,
        "atr": atr,
        "top_bids": top_bids,
        "top_asks": top_asks
    }

    return {
        "label": label,
        "icon": icon,
        "color": color,
        "confidence_pct": confidence_pct,
        "recommendation": recommendation,
        "strength": strength,
        "entry": entry,
        "target": target,
        "stop": stop,
        "metrics": metrics,
        "weights": {"funding":0.10,"oi":0.15,"cvd":0.20,"orderbook":0.25,"backtest":0.30},
        "raw": raw,
        "reason": reason
    }

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Smart Money Scanner V3.8", layout="wide")
st.title("🧠 Smart Money Scanner V3.8 — Integrated Signals + Trade Suggestion (OKX)")

# Sidebar inputs
inst_type = st.sidebar.selectbox("Instrument Type", ["SWAP","SPOT"])
instruments = fetch_instruments(inst_type)
if not instruments:
    st.sidebar.error("Unable to load instruments from OKX. تحقق من الاتصال.")
    st.stop()

default_inst = "BTC-USDT-SWAP" if inst_type=="SWAP" else instruments[0]
instId = st.sidebar.selectbox("Instrument", instruments, index=instruments.index(default_inst) if default_inst in instruments else 0)
bar = st.sidebar.selectbox("Timeframe", ["5m","15m","1H","4H","1D"], index=2)
show_raw = st.sidebar.checkbox("Show Raw metrics (Advanced)", value=False)

if st.sidebar.button("🚀 Run Analysis"):
    with st.spinner("Fetching & computing…"):
        result = compute_confidence(instId, bar)

    # Header summary
    st.subheader(f"{result['label']} — Confidence: {result['confidence_pct']}%")
    st.markdown(
        f"<h4 style='color:{result['color']}; margin-top:0'>"
        f"{result['icon']} {result['recommendation']} — {result['strength']}"
        f"</h4>",
        unsafe_allow_html=True
    )

    # Times
    st.caption(f"Last Updated (Local – Africa/Algiers): {local_now_str()}  |  UTC: {utc_now_str()}")

    # Live price
    st.metric("💵 Live Price", format_price(result["raw"]["price"]))

    # Metrics cards
    st.markdown("### 📊 Core Metrics")
    icons = {"funding":"💰","oi":"📊","cvd":"📈","orderbook":"⚖️","backtest":"🧪"}
    cols = st.columns(5)
    order = ["funding","oi","cvd","orderbook","backtest"]
    for i, k in enumerate(order):
        score = result["metrics"][k]
        weight = result["weights"][k]
        contrib = round(score*weight*100, 2)
        with cols[i]:
            st.metric(label=f"{icons[k]} {k.upper()}", value=f"{score:.3f}", delta=f"w={weight}")
            st.caption(f"Contribution: {contrib}%")

    # Trade plan
    st.markdown("---")
    st.markdown("### 📝 Trade Plan")
    st.write("**Why:**", result["reason"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Entry", format_price(result["entry"]))
    c2.metric("Target", format_price(result["target"]) if result["target"] else "N/A")
    c3.metric("Stop", format_price(result["stop"]) if result["stop"] else "N/A")

    # Extra info (no charts)
    st.markdown("---")
    st.markdown("### 🔍 Structure & Orderflow")
    support = result["raw"]["support"]
    resistance = result["raw"]["resistance"]
    st.write(f"• **Support (approx):** {format_price(support)}  |  **Resistance (approx):** {format_price(resistance)}")
    st.write(f"• **Candle signal:** {result['raw']['candle_signal'] or 'None'}")
    # compact liquidity snapshot
    tb = result["raw"]["top_bids"] or []
    ta = result["raw"]["top_asks"] or []
    if tb or ta:
        st.write("• **Liquidity (near price):**")
        if tb:
            st.write("  - Top Bids:", ", ".join([f"{format_price(x['price'])} ({format_num(x['size'],2)})" for x in tb]))
        if ta:
            st.write("  - Top Asks:", ", ".join([f"{format_price(x['price'])} ({format_num(x['size'],2)})" for x in ta]))

    # Raw metrics (optional)
    if show_raw:
        st.markdown("---")
        st.markdown("### 📂 Raw Metrics (Advanced)")
        st.json(result["raw"])

else:
    st.info("اختر الأداة والإطار الزمني من الشريط الجانبي ثم اضغط “🚀 Run Analysis”.")
