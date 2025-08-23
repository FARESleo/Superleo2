import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from math import isnan
from datetime import datetime

OKX_BASE = "https://www.okx.com"

# ----------------------------
# CSS Styling for high-contrast minimal design
# ----------------------------
st.markdown("""
<style>
/* Main app container */
.css-18e3th9 {
    padding-top: 1rem;
    padding-bottom: 5rem;
    padding-left: 1rem;
    padding-right: 1rem;
    background-color: #000000; /* Pure black background for max contrast */
    color: #FFFFFF; /* Pure white text */
}

/* Hide Streamlit's default hamburger menu */
#MainMenu {visibility: hidden;}

/* Style the header and buttons */
.css-1av0ku6, .css-1y480o3, .css-1d374r {
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
}

/* Card-like containers for sections */
.stCard {
    background-color: rgba(26, 26, 26, 0.7); /* Semi-transparent background */
    backdrop-filter: blur(10px); /* Frosted glass effect */
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.1); /* Subtle white border */
}

/* Metrics styling */
.stMetric {
    background-color: rgba(255, 255, 255, 0.05); /* Very subtle gray for metrics */
    border-radius: 8px;
    padding: 10px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: transform 0.2s ease-in-out, border-color 0.2s ease-in-out;
}
.stMetric:hover {
    transform: translateY(-3px);
    border: 1px solid #007BFF; /* Blue glow on hover */
}
.stMetric .st-bd .st-b5 {
    color: #FFFFFF !important; /* Make metric value pure white */
    font-size: 2em;
    font-weight: 700;
}
.stMetric .st-bd .st-at {
    color: #B0B0B0 !important; /* Light gray for labels */
}

/* Color coding for metrics */
.stMetric.bullish {
    border-left: 5px solid #28A745; /* Bright green border */
}
.stMetric.bearish {
    border-left: 5px solid #DC3545; /* Bright red border */
}
.stMetric.neutral {
    border-left: 5px solid #6c757d; /* Gray for neutrality */
}

/* Font and text color */
html, body, .st-emotion-cache-12oz5g7, .st-emotion-cache-1r651o3 {
    color: #FFFFFF !important;
    font-family: 'Segoe UI', 'Roboto', sans-serif;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #007BFF; /* Blue for headings */
    border-bottom: 2px solid #007BFF;
    padding-bottom: 5px;
    margin-bottom: 15px;
}

/* Buttons */
.stButton>button {
    background-color: #007BFF;
    color: #FFFFFF;
    font-weight: bold;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    transition: background-color 0.2s ease-in-out;
}
.stButton>button:hover {
    background-color: #0056b3;
}

/* Special styles for main headers */
.st-emotion-cache-1d374r {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.st-emotion-cache-1d374r h1 {
    border-bottom: none;
    margin-bottom: 0;
}
</style>
""", unsafe_allow_html=True)

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

def find_liquidity_zones(bids_df, asks_df, price_range_pct=0.01):
    if bids_df is None or asks_df is None or bids_df.empty or asks_df.empty:
        return None, None
    
    current_price = (bids_df["price"].iloc[0] + asks_df["price"].iloc[0]) / 2
    
    bid_zones = bids_df[bids_df["price"] > current_price * (1 - price_range_pct)]
    ask_zones = asks_df[asks_df["price"] < current_price * (1 + price_range_pct)]
    
    top_bids = bid_zones.nlargest(5, "size").to_dict('records')
    top_asks = ask_zones.nlargest(5, "size").to_dict('records')
    
    return top_bids, top_asks

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

def detect_candle_signal(ohlcv_df, bar):
    if ohlcv_df.empty or len(ohlcv_df) < 3:
        return None
    
    last = ohlcv_df.iloc[-1]
    prev = ohlcv_df.iloc[-2]
    
    if last["c"] > last["o"] and prev["c"] < prev["o"] and last["c"] > prev["o"] and last["o"] < prev["c"]:
        return "Bullish Engulfing"
    if last["c"] < last["o"] and prev["c"] > prev["o"] and last["c"] < prev["o"] and last["o"] > prev["c"]:
        return "Bearish Engulfing"

    if bar in ["5m", "15m", "1H"]:
        prev2 = ohlcv_df.iloc[-3]
        is_bearish_prev2 = prev2["c"] < prev2["o"]
        is_small_body_prev = abs(prev["o"] - prev["c"]) < ((last["h"] - last["l"]) * 0.2)
        is_bullish_last = last["c"] > last["o"]
        
        if is_bearish_prev2 and is_small_body_prev and is_bullish_last:
            return "Bullish Morning Star"
            
    return None

def calculate_atr(ohlcv_df, period=14):
    if ohlcv_df.empty or len(ohlcv_df) < period:
        return None
    df = ohlcv_df.copy()
    high = df['h']
    low = df['l']
    close = df['c']
    
    df['tr1'] = high - low
    df['tr2'] = abs(high - close.shift(1))
    df['tr3'] = abs(low - close.shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    atr = df['tr'].rolling(period).mean().iloc[-1]
    return atr


def compute_confidence(instId, bar="1H"):
    with st.spinner("Computing — gathering live data..."):
        ohlcv = fetch_ohlcv(instId, bar, limit=300)
        price = fetch_ticker(instId)
        funding = fetch_funding(instId)
        oi = fetch_oi(instId)
        bids, asks = fetch_orderbook(instId, depth=60)
        trades = fetch_trades(instId, limit=400)
    
    cvd = compute_cvd(trades)
    ob_imb = orderbook_imbalance(bids, asks)
    top_bids, top_asks = find_liquidity_zones(bids, asks)
    bt_win = simple_backtest_winrate(ohlcv, lookahead=6, stop_pct=0.01, rr=2.0)
    support, resistance = compute_support_resistance(ohlcv)
    candle_signal = detect_candle_signal(ohlcv, bar)

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
    
    atr = calculate_atr(ohlcv, period=14)
    if atr is None or price is None:
        label = "⚠️ Neutral"
        recommendation = "Wait"
        entry = price
        target = stop = None
        strength = "N/A"
        reason = "بيانات غير كافية."
    
    else:
        # Define signal strengths
        is_bullish_strong = (
            (confidence_pct >= 65) and
            (cvd is not None and cvd > 0) and
            (ob_imb is not None and ob_imb > 0) and
            (candle_signal in ["Bullish Engulfing", "Bullish Morning Star"])
        )
        
        is_bullish_weak = (
            (confidence_pct >= 50) and
            ((cvd is not None and cvd > 0) or (candle_signal in ["Bullish Engulfing", "Bullish Morning Star"]))
        )
        
        is_bearish_strong = (
            (confidence_pct <= 35) and
            (cvd is not None and cvd < 0) and
            (ob_imb is not None and ob_imb < 0) and
            (candle_signal == "Bearish Engulfing")
        )

        is_bearish_weak = (
            (confidence_pct <= 50) and
            ((cvd is not None and cvd < 0) or (candle_signal == "Bearish Engulfing"))
        )
        
        if is_bullish_strong:
            label = "📈 Bullish"
            recommendation = "LONG"
            strength = "Strong"
            entry = price
            target = round(entry + (atr * 2), 6)
            stop = round(entry - atr, 6)
            reason = f"إشارة صعودية قوية: {candle_signal} + CVD إيجابي + سجل طلبات صاعد."
        elif is_bullish_weak:
            label = "📈 Bullish"
            recommendation = "LONG"
            strength = "Weak"
            entry = price
            target = round(entry + (atr * 1.5), 6)
            stop = round(entry - atr, 6)
            reason = f"إشارة صعودية ضعيفة: {candle_signal} أو CVD إيجابي، لكن الإشارات مختلطة."
        elif is_bearish_strong:
            label = "📉 Bearish"
            recommendation = "SHORT"
            strength = "Strong"
            entry = price
            target = round(entry - (atr * 2), 6)
            stop = round(entry + atr, 6)
            reason = f"إشارة هبوطية قوية: {candle_signal} + CVD سلبي + سجل طلبات هابط."
        elif is_bearish_weak:
            label = "📉 Bearish"
            recommendation = "SHORT"
            strength = "Weak"
            entry = price
            target = round(entry - (atr * 1.5), 6)
            stop = round(entry + atr, 6)
            reason = f"إشارة هبوطية ضعيفة: {candle_signal} أو CVD سلبي، لكن الإشارات مختلطة."
        else:
            label = "⚠️ Neutral"
            recommendation = "Wait"
            strength = "Neutral"
            entry = price
            target = stop = None
            reason = "لا يوجد سبب مقنع للدخول. المؤشرات متضاربة."

    raw = {"price":price,"funding":funding,"oi":oi,"cvd":cvd,"orderbook_imbalance":ob_imb,"backtest_win":bt_win,"support":support,"resistance":resistance,"candle_signal":candle_signal, "top_bids":top_bids, "top_asks":top_asks, "atr":atr}

    return {"label":label,"confidence_pct":confidence_pct,"recommendation":recommendation,"strength":strength,"entry":entry,"target":target,"stop":stop,"metrics":metrics,"weights":weights,"raw":raw,"reason":reason}

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Smart Money Scanner", layout="wide")

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'selected_instId' not in st.session_state:
    st.session_state.selected_instId = "BTC-USDT-SWAP"
if 'bar' not in st.session_state:
    st.session_state.bar = "1H"

# Fetch all instruments once
all_instruments = fetch_instruments("SWAP") + fetch_instruments("SPOT")
if not all_instruments:
    st.error("Unable to load instruments from OKX.")
    st.stop()
    
# Title and Button in the same row
header_col1, header_col2 = st.columns([0.7, 0.3])
with header_col1:
    st.markdown("<h1><span style='font-size: 2.2rem;'>🧠 Smart Money Scanner</span></h1>", unsafe_allow_html=True)

def run_analysis_clicked():
    st.session_state.analysis_results = compute_confidence(st.session_state.selected_instId, st.session_state.bar)

with header_col2:
    if st.button("Get Analysis"):
        run_analysis_clicked()
        
# User inputs
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.session_state.selected_instId = st.selectbox(
        "Select Instrument",
        all_instruments,
        index=all_instruments.index(st.session_state.selected_instId) if st.session_state.selected_instId in all_instruments else 0
    )
with col2:
    st.session_state.bar = st.selectbox(
        "Timeframe",
        ["5m", "15m", "1H", "6H", "12H"],
        index=["5m", "15m", "1H", "6H", "12H"].index(st.session_state.bar) if st.session_state.bar in ["5m", "15m", "1H", "6H", "12H"] else 2
    )
st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Display results if available
if st.session_state.analysis_results:
    result = st.session_state.analysis_results
    
    st.markdown("---")

    # Main metrics with color coding
    color_class = ""
    if "Bullish" in result['label']:
        color_class = "bullish"
    elif "Bearish" in result['label']:
        color_class = "bearish"
    else:
        color_class = "neutral"

    st.markdown(f"<div class='stCard'>", unsafe_allow_html=True)
    st.markdown("### 📈 Analysis Overview")
    main_col1, main_col2, main_col3 = st.columns(3)
    
    with main_col1:
        st.markdown(f"<div class='stMetric {color_class}'>", unsafe_allow_html=True)
        st.metric(label=f"Confidence", value=f"{result['confidence_pct']}%", delta=result['label'])
        st.markdown("</div>", unsafe_allow_html=True)
    
    with main_col2:
        st.markdown(f"<div class='stMetric'>", unsafe_allow_html=True)
        st.metric(label="Recommendation", value=f"{result['recommendation']} ({result['strength']})")
        st.markdown("</div>", unsafe_allow_html=True)

    with main_col3:
        st.markdown(f"<div class='stMetric'>", unsafe_allow_html=True)
        st.metric(label="Live Price", value=f"{result['raw']['price']:,}" if result['raw']['price'] else "N/A")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Core Metrics
    st.markdown(f"<div class='stCard'>", unsafe_allow_html=True)
    st.markdown("### 📊 Core Metrics")
    icons = {"funding":"💰","oi":"📊","cvd":"📈","orderbook":"⚖️","backtest":"🧪"}
    cols = st.columns(5)
    for idx, k in enumerate(["funding","oi","cvd","orderbook","backtest"]):
        col = cols[idx]
        score = result["metrics"][k]
        weight = result["weights"][k]
        contrib = round(score*weight*100,2)
        with col:
            st.markdown(f"<div class='stMetric'>", unsafe_allow_html=True)
            st.metric(label=f"{icons[k]} {k.upper()}", value=f"{score:.3f}", delta=f"w={weight}")
            st.caption(f"Contribution: {contrib}%")
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Trade Plan
    st.markdown(f"<div class='stCard'>", unsafe_allow_html=True)
    st.markdown("### 📝 Trade Plan")
    st.markdown(f"**Reason:** {result['reason']}")
    trade_col1, trade_col2, trade_col3 = st.columns(3)
    trade_col1.metric("Entry Price", f"{result['entry']:,}" if result['entry'] else "N/A")
    trade_col2.metric("Target Price", f"{result['target']:,}" if result['target'] else "N/A")
    trade_col3.metric("Stop Loss", f"{result['stop']:,}" if result['stop'] else "N/A")
    st.markdown("</div>", unsafe_allow_html=True)

    # Additional Analysis
    st.markdown(f"<div class='stCard'>", unsafe_allow_html=True)
    st.markdown("### 🔍 Additional Analysis")
    st.markdown(f"• **Support:** {result['raw']['support']:,} | **Resistance:** {result['raw']['resistance']:,}")
    st.markdown(f"• **Candle Signal:** {result['raw']['candle_signal'] if result['raw']['candle_signal'] else 'None'}")
    st.markdown("</div>", unsafe_allow_html=True)

    show_raw = st.checkbox("Show Raw metrics", value=False)
    if show_raw:
        st.markdown("### Raw metrics (for transparency)")
        st.json(result["raw"])

else:
    st.info("Select instrument/timeframe and press 'Get Analysis' to begin.")
