import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from math import isnan
from datetime import datetime

# --- CSS للتصميم الاحترافي والأزرار الجديدة ---
st.markdown(
    """
    <style>
    /* خلفية التطبيق */
    .stApp {
        background-image: url("https://i.imgur.com/Utvjk6E.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* الشريط الثابت في الأسفل */
    .bottom-nav {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #262730;
        padding: 10px 0;
        display: flex;
        justify-content: space-around;
        align-items: center;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.3);
        z-index: 1000;
    }
    .st-emotion-cache-1pxx35k.e1f1d6gn2 { /* هذه الفئة تستهدف الأزرار في الأعمدة */
        display: flex !important;
        flex-direction: column;
        align-items: center;
        color: white;
        background-color: transparent;
        border: none;
        padding: 0;
        margin: 0;
    }

    .st-emotion-cache-1pxx35k.e1f1d6gn2:hover {
        color: #6A11CB;
    }
    
    .st-emotion-cache-1pxx35k.e1f1d6gn2[data-active=true] {
        color: #6A11CB;
    }

    /* بقية الأنماط القديمة */
    .custom-card {
        background-color: #F8F8F8;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: #333;
    }
    .card-header {
        font-size: 14px;
        color: #777;
        text-transform: uppercase;
        font-weight: bold;
    }
    .card-value {
        font-size: 28px;
        font-weight: bold;
        margin-top: 5px;
    }
    .progress-bar-container {
        background-color: #ddd;
        border-radius: 50px;
        height: 10px;
        width: 100%;
        margin-top: 10px;
    }
    .progress-bar {
        height: 100%;
        border-radius: 50px;
        transition: width 0.5s ease-in-out;
    }
    .trade-plan-card {
        background-color: #f0f0f0;
        border-left: 5px solid #6A11CB;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .trade-plan-title {
        font-size: 24px;
        font-weight: bold;
        color: #333;
        margin-bottom: 15px;
    }
    .trade-plan-metric {
        margin-bottom: 15px;
    }
    .trade-plan-metric-label {
        font-size: 16px;
        color: #555;
        font-weight: bold;
    }
    .trade-plan-metric-value {
        font-size: 20px;
        font-weight: bold;
    }
    .reason-card {
        background-color: #f0f4f7;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 4px solid;
    }
    .reason-text {
        font-size: 16px;
        line-height: 1.6;
        margin-top: 5px;
        font-style: italic;
    }
    .reason-card.bullish {
        border-color: #4CAF50;
        background-color: #f0fbf0;
        color: #2e7d32;
    }
    .reason-card.bearish {
        border-color: #d32f2f;
        background-color: #fff0f0;
        color: #b71c1c;
    }
    .reason-card.neutral {
        border-color: #ff9800;
        background-color: #fff8f0;
        color: #e65100;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- كود الأيقونات (مكتبة Font Awesome) ---
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">', unsafe_allow_html=True)


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

# Helper function to format prices
def format_price(price, decimals=None):
    if price is None or isnan(price):
        return "N/A"
    if decimals is None:
        if price >= 1000: decimals = 2
        elif price >= 10: decimals = 3
        else: decimals = 4
    return f"{price:,.{decimals}f}"

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'selected_instId' not in st.session_state:
    st.session_state.selected_instId = "BTC-USDT-SWAP"
if 'bar' not in st.session_state:
    st.session_state.bar = "1H"
if 'page' not in st.session_state:
    st.session_state.page = 'main_scanner'

# Fetch all instruments once
all_instruments = fetch_instruments("SWAP") + fetch_instruments("SPOT")
if not all_instruments:
    st.error("Unable to load instruments from OKX.")
    st.stop()
    
# Title and button placeholders (we will hide these later)
st.header("🧠 Smart Money Scanner")
st.markdown("---")

# Main content based on page state
if st.session_state.page == 'main_scanner':
    # User inputs for scanner
    st.session_state.selected_instId = st.selectbox("Select Instrument", all_instruments, index=all_instruments.index(st.session_state.selected_instId) if st.session_state.selected_instId in all_instruments else 0)
    st.session_state.bar = st.selectbox("Timeframe", ["30m", "15m", "1H", "6H", "12H"], index=["30m", "15m", "1H", "6H", "12H"].index(st.session_state.bar) if st.session_state.bar in ["30m", "15m", "1H", "6H", "12H"] else 0)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Run analysis button
    st.button("Run Analysis", on_click=lambda: st.session_state.update(analysis_results=compute_confidence(st.session_state.selected_instId, st.session_state.bar)))
        
    # Display results
    if st.session_state.analysis_results:
        result = st.session_state.analysis_results
        
        # Get the confidence color
        def get_confidence_color(pct):
            if pct <= 40: return "red"
            if pct <= 60: return "orange"
            else: return "green"

        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown(f"### Current Price: ${format_price(result['raw']['price'])}", unsafe_allow_html=True)
        
        # Main results section
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div class="custom-card">
                    <div class="card-header">Confidence Score</div>
                    <div class="card-value" style="color:{get_confidence_color(result['confidence_pct'])}">{result['confidence_pct']}%</div>
                    <div class="progress-bar-container">
                        <div class="progress-bar" style="width: {result['confidence_pct']}%; background-color: {get_confidence_color(result['confidence_pct'])};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div class="trade-plan-card">
                    <div class="trade-plan-title">Recommendation: {result['recommendation']}</div>
                    <div class="trade-plan-metric">
                        <div class="trade-plan-metric-label">Entry:</div>
                        <div class="trade-plan-metric-value">{format_price(result['entry'])}</div>
                    </div>
                    <div class="trade-plan-metric">
                        <div class="trade-plan-metric-label">Target:</div>
                        <div class="trade-plan-metric-value">{format_price(result['target'])}</div>
                    </div>
                    <div class="trade-plan-metric">
                        <div class="trade-plan-metric-label">Stop Loss:</div>
                        <div class="trade-plan-metric-value">{format_price(result['stop'])}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
        
        # Reason Card
        reason_class = "bullish" if result["recommendation"] == "LONG" else ("bearish" if result["recommendation"] == "SHORT" else "neutral")
        st.markdown(f"""
        <div class="reason-card {reason_class}">
            <div class="reason-text">**Reason:** {result['reason']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Key Metrics")
        st.markdown(f"**CVD (Cumulative Volume Delta):** {format_price(result['raw']['cvd'], decimals=2) if result['raw']['cvd'] is not None else 'N/A'}")
        st.markdown(f"**Orderbook Imbalance:** {round(result['raw']['orderbook_imbalance']*100, 2) if result['raw']['orderbook_imbalance'] is not None else 'N/A'}%")
        st.markdown(f"**Funding Rate:** {round(result['raw']['funding']*100, 4) if result['raw']['funding'] is not None else 'N/A'}%")
        st.markdown(f"**Backtest Win Rate:** {round(result['raw']['backtest_win']*100, 2) if result['raw']['backtest_win'] is not None else 'N/A'}%")
        st.markdown(f"**ATR (14-period):** {format_price(result['raw']['atr'], decimals=4) if result['raw']['atr'] is not None else 'N/A'}")
        st.markdown(f"**Candle Signal:** {result['raw']['candle_signal'] if result['raw']['candle_signal'] is not None else 'N/A'}")
        
elif st.session_state.page == 'calculator':
    st.title("Risk Calculator")
    st.info("Here you can build your risk calculator.")
    
elif st.session_state.page == 'tracker':
    st.title("Trade Tracker")
    st.info("Here you can build your trade tracker.")
    
# --- شريط التنقل الثابت في الأسفل ---
st.markdown('<div class="bottom-nav">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.button("Scanner", key="btn_scanner", on_click=lambda: st.session_state.update(page='main_scanner'))
with col2:
    st.button("Calculator", key="btn_calculator", on_click=lambda: st.session_state.update(page='calculator'))
with col3:
    st.button("Tracker", key="btn_tracker", on_click=lambda: st.session_state.update(page='tracker'))
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<style>
.st-emotion-cache-1pxx35k.e1f1d6gn2 > div {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
}
.st-emotion-cache-1pxx35k.e1f1d6gn2 > div > span {
    display: none;
}
.st-emotion-cache-1pxx35k.e1f1d6gn2:before {
    font-family: 'Font Awesome 6 Free';
    font-weight: 900;
    font-size: 24px;
    margin-bottom: 5px;
}
[data-testid="stButton-btn_scanner"]:before {
    content: '\\f202'; /* الأيقونة: fa-brain */
}
[data-testid="stButton-btn_calculator"]:before {
    content: '\\f1ec'; /* الأيقونة: fa-calculator */
}
[data-testid="stButton-btn_tracker"]:before {
    content: '\\f201'; /* الأيقونة: fa-chart-line */
}
</style>
""", unsafe_allow_html=True)
