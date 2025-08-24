import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from math import isnan
from datetime import datetime

# --- كود CSS لتصميم الواجهة ---
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://i.imgur.com/Utvjk6E.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        z-index: -1;
    }
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
    
    /* --- New CSS for Bottom Navigation Bar --- */
    .bottom-navbar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        background-color: white; 
        box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
        padding: 10px 20px;
        display: flex;
        justify-content: space-around;
        align-items: center;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
    }
    
    .bottom-navbar .st-cr .st-cv {
        flex-direction: row;
        justify-content: space-around;
        gap: 15px;
    }
    
    .bottom-navbar .st-cr .st-cv .st-ce,
    .bottom-navbar .st-cr .st-cv .st-cf {
        border-radius: 50px;
        padding: 10px 20px;
        color: #6A11CB; 
        background-color: #f0f0f0;
        transition: all 0.2s ease;
    }
    
    .bottom-navbar .st-cr .st-cv .st-ce[data-selected="true"] {
        background-image: linear-gradient(to right, #6A11CB, #2575FC);
        color: white;
        transform: translateY(-2px);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# --- الكود الأصلي للدوال والمكونات (بدون تغيير) ---
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

# New function to calculate PnL percentages
def calculate_pnl_percentages(entry_price, take_profit, stop_loss):
    if entry_price is None or take_profit is None or stop_loss is None or entry_price == 0:
        return None, None
    
    profit_pct = ((take_profit - entry_price) / entry_price) * 100
    loss_pct = ((stop_loss - entry_price) / entry_price) * 100
    
    # Correct for long vs short
    is_long = take_profit > entry_price
    if not is_long:
        profit_pct, loss_pct = loss_pct, profit_pct
    
    return profit_pct, loss_pct

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
    
# --- الشريط العلوي مع الأزرار ---
st.markdown("<h1 style='font-size: 2.5rem; font-weight: bold; margin: 0;'>🧠 Smart Money Scanner</h1>", unsafe_allow_html=True)
header_col1, header_col2 = st.columns([1, 2])

def run_analysis_clicked():
    st.session_state.analysis_results = compute_confidence(st.session_state.selected_instId, st.session_state.bar)
    
with header_col1:
    if st.button("🚀 Go"):
        run_analysis_clicked()

with header_col2:
    # Display last updated time and user inputs
    st.markdown(f"**آخر تحديث:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("---")

st.session_state.selected_instId = st.selectbox("حدد الأداة", all_instruments, index=all_instruments.index(st.session_state.selected_instId) if st.session_state.selected_instId in all_instruments else 0)
st.session_state.bar = st.selectbox("الإطار الزمني", ["30m", "15m", "1H", "6H", "12H"], index=["30m", "15m", "1H", "6H", "12H"].index(st.session_state.bar) if st.session_state.bar in ["30m", "15m", "1H", "6H", "12H"] else 0)

# ----------------------------------------------------
# 🌟 الدوال لكل واجهة (بدون تغيير) 🌟
# ----------------------------------------------------

def trading_calculator_app():
    """
    Creates and manages the trading calculator UI using pure Streamlit.
    """
    st.header("🧮 حاسبة التداول")
    
    imr = st.number_input("الهامش المبدئي (IMR %)", min_value=0.01, value=2.0, step=0.1, help="النسبة المئوية من إجمالي قيمة الصفقة التي تودعها.")
    mmr = st.number_input("هامش الحِفاظ (MMR %)", min_value=0.01, value=1.0, step=0.1, help="الحد الأدنى من الهامش المطلوب للحفاظ على الصفقة.")
    
    if imr > mmr and imr > 0:
        margin_diff = imr - mmr
        max_leverage = round(100 / margin_diff, 2)
        leverage_options = {
            "5x (منخفضة)": 5,
            "10x (منخفضة)": 10,
            "20x (متوسطة)": 20,
            "30x (متوسطة)": 30,
            "50x (عالية)": 50,
            "100x (عالية جداً)": 100,
            f"{max_leverage}x (القصوى)": max_leverage
        }
        
        valid_leverage_options = {k: v for k, v in leverage_options.items() if v <= max_leverage}
        
        st.markdown("**اختر الرافعة المالية**")
        leverage_cols = st.columns(len(valid_leverage_options))
        
        for i, (label, value) in enumerate(valid_leverage_options.items()):
            if leverage_cols[i].button(label, key=f"lev_btn_{value}", use_container_width=True):
                st.session_state.selected_leverage = value
    else:
        st.warning("الرجاء إدخال قيم صالحة للهامش المبدئي وهامش الحفاظ.")
        st.session_state.selected_leverage = None
        st.session_state.selected_leverage = None
        
    col1, col2 = st.columns(2)
    with col1:
        capital = st.number_input("المبلغ (USDT)", min_value=0.01, value=10.0, step=0.1)
    with col2:
        current_price = st.number_input("السعر الحالي", min_value=0.01, value=25000.0, step=0.01)

    col3, col4 = st.columns(2)
    with col3:
        target_price = st.number_input("سعر الهدف", min_value=0.01, value=26000.0, step=0.01)
    with col4:
        direction = st.selectbox("الاتجاه", ["📈 شراء (Long)", "📉 بيع (Short)"])
    
    if 'selected_leverage' in st.session_state and st.session_state.selected_leverage and capital and current_price and target_price and imr and mmr:
        leverage = st.session_state.selected_leverage
        is_long = direction == "📈 شراء (Long)"
        
        margin_diff = imr - mmr
        price_change_pct = ((target_price - current_price) / current_price) * 100
        actual_price_change = price_change_pct if is_long else -price_change_pct
        roi_percent = actual_price_change * leverage
        pnl_value = (capital * roi_percent) / 100
        
        if is_long:
            liquidation_price = current_price * (1 - (margin_diff / 100))
        else:
            liquidation_price = current_price * (1 + (margin_diff / 100))
            
        st.markdown("---")
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric(label="فرق الهامش", value=f"{margin_diff:.2f}%")
        with metrics_col2:
            st.metric(label="التغير %", value=f"{actual_price_change:.2f}%")
        with metrics_col3:
            st.metric(label="ROI %", value=f"{roi_percent:.2f}%")
            
        metrics_col4, metrics_col5 = st.columns(2)
        with metrics_col4:
            pnl_color = "green" if pnl_value >= 0 else "red"
            st.markdown(f"**<p style='color: {pnl_color}; font-size: 1.5rem;'>PnL: {pnl_value:.2f} USDT</p>**", unsafe_allow_html=True)
        with metrics_col5:
            liq_color = "red"
            st.markdown(f"**<p style='color: {liq_color}; font-size: 1.5rem;'>سعر التصفية: {liquidation_price:.4f}</p>**", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def get_live_market_data():
    """Fetches live market data from CoinGecko API."""
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


def live_market_tracker():
    """Renders the Live Market Tracker UI."""
    st.markdown("---")
    st.header("📈 متتبع السوق اللحظي")
    st.write("استكشف العملات ذات التغيرات السعرية الكبيرة خلال الـ 24 ساعة الماضية.")

    # UI Controls
    cols = st.columns(3)
    with cols[0]:
        threshold = st.selectbox("عتبة التغيير (%):", options=[1, 5, 10, 20, 50, 100], index=4)
    with cols[1]:
        search_query = st.text_input("ابحث عن عملة...").lower()
    with cols[2]:
        filter_type = st.selectbox("الفلتر:", options=["الكل", "صعود فقط", "هبوط فقط"], format_func=lambda x: x)

    # Fetch and filter data
    with st.spinner("جارٍ تحديث بيانات السوق اللحظية..."):
        all_coins_df = get_live_market_data()

    if all_coins_df.empty:
        st.warning("تعذر جلب البيانات. يرجى المحاولة لاحقاً.")
        return

    # Filter logic
    filtered_df = all_coins_df[all_coins_df['price_change_percentage_24h'].notna()].copy()
    filtered_df['price_change_abs'] = filtered_df['price_change_percentage_24h'].abs()
    filtered_df = filtered_df[filtered_df['price_change_abs'] >= threshold]

    if filter_type == "صعود فقط":
        filtered_df = filtered_df[filtered_df['price_change_percentage_24h'] > 0]
    elif filter_type == "هبوط فقط":
        filtered_df = filtered_df[filtered_df['price_change_percentage_24h'] < 0]

    if search_query:
        filtered_df = filtered_df[filtered_df['name'].str.lower().str.contains(search_query) | 
                                 filtered_df['symbol'].str.lower().str.contains(search_query)]
    
    # Use only the symbol
    filtered_df['رمز العملة'] = filtered_df['symbol'].str.upper()
    
    # Sort by price change
    filtered_df = filtered_df.sort_values(by='price_change_abs', ascending=False)
    
    if filtered_df.empty:
        st.info("لا توجد عملات مطابقة للمعايير المحددة.")
    else:
        # Prepare data for display
        display_df = filtered_df[[
            'رمز العملة',
            'current_price',
            'price_change_percentage_24h',
            'high_24h',
            'low_24h'
        ]].rename(columns={
            'رمز العملة': 'الرمز',
            'current_price': 'السعر ($)',
            'price_change_percentage_24h': 'التغيير (24س) %',
            'high_24h': 'أعلى سعر (24س) ($)',
            'low_24h': 'أدنى سعر (24س) ($)'
        })
        
        # Format columns for better readability
        display_df['السعر ($)'] = display_df['السعر ($)'].apply(lambda x: f"{x:,.4f}" if x > 0.001 else f"{x:,.8f}")
        display_df['التغيير (24س) %'] = display_df['التغيير (24س) %'].apply(lambda x: f"{x:,.2f}%")
        display_df['أعلى سعر (24س) ($)'] = display_df['أعلى سعر (24س) ($)'].apply(lambda x: f"{x:,.4f}" if x > 0.001 else f"{x:,.8f}")
        display_df['أدنى سعر (24س) ($)'] = display_df['أدنى سعر (24س) ($)'].apply(lambda x: f"{x:,.4f}" if x > 0.001 else f"{x:,.8f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption(f"آخر تحديث: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- عرض الصفحة المحددة بناءً على شريط التنقل السفلي ---
if 'selected_leverage' not in st.session_state:
    st.session_state.selected_leverage = None

st.markdown('<div class="bottom-navbar">', unsafe_allow_html=True)
selected_page = st.radio(
    "Go to",
    ["التحليل", "الحاسبة", "المتتبع"],
    horizontal=True,
    label_visibility="collapsed",
    key="bottom_nav"
)
st.markdown('</div>', unsafe_allow_html=True)

if selected_page == "التحليل":
    if st.session_state.analysis_results:
        result = st.session_state.analysis_results
    
        # Get the confidence color based on the percentage
        def get_confidence_color(pct):
            if pct <= 40: return "red"
            if pct <= 60: return "orange"
            return "green"

        confidence_color = get_confidence_color(result['confidence_pct'])
        progress_width = result['confidence_pct']

        # Get the correct emoji for the recommendation
        rec_emoji = ""
        if result['recommendation'] == "LONG":
            rec_emoji = "🚀"
        elif result['recommendation'] == "SHORT":
            rec_emoji = "🔻"
        else:
            rec_emoji = "⏳"
        
        # Visual alert system
        if result['confidence_pct'] >= 80:
            st.balloons()
            st.success("🎉 إشارة قوية جدًا تم اكتشافها! انتبه لهذه الفرصة.", icon="🔥")
        elif result['confidence_pct'] <= 20:
            st.warning("⚠️ إشارة ضعيفة جدًا. يفضل توخي الحذر.")
            
        # Display the main metrics in cards
        cols = st.columns(3)
        
        with cols[0]:
            st.markdown(f"""
                <div class="custom-card">
                    <div class="card-header">📊 الثقة</div>
                    <div class="card-value">{result['confidence_pct']}%</div>
                    <div class="progress-bar-container">
                        <div class="progress-bar" style="width:{progress_width}%; background-color:{confidence_color};"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with cols[1]:
            st.markdown(f"""
                <div class="custom-card">
                    <div class="card-header">⭐ التوصية</div>
                    <div class="card-value">{rec_emoji} {result['recommendation']}</div>
                    <div style="font-size: 14px; color: #999;">({result['strength']})</div>
                </div>
            """, unsafe_allow_html=True)
            
        with cols[2]:
            st.markdown(f"""
                <div class="custom-card">
                    <div class="card-header">📈 السعر الحالي</div>
                    <div class="card-value">{format_price(result['raw']['price'])}</div>
                    <div style="font-size: 14px; color: #999;">{st.session_state.selected_instId}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # The new, improved Trade Plan section
        reason_class = "neutral"
        if "صعودية" in result['reason']:
            reason_class = "bullish"
        elif "هبوطية" in result['reason']:
            reason_class = "bearish"

        st.markdown(f"""
            <div class="trade-plan-card">
                <div class="trade-plan-title">📝 Trade Plan</div>
        """, unsafe_allow_html=True)
        
        trade_plan_col1, trade_plan_col2 = st.columns([2, 1])
        
        with trade_plan_col1:
            st.markdown(f"""
                <div class="reason-card {reason_class}">
                    <div class="trade-plan-metric-label">السبب:</div>
                    <div class="reason-text">{result['reason']}</div>
                </div>
            """, unsafe_allow_html=True)
            
        with trade_plan_col2:
            # Calculate PnL percentages
            profit_pct, loss_pct = calculate_pnl_percentages(
                result['entry'], 
                result['target'], 
                result['stop']
            )
            
            st.markdown(f"""
                <div class="trade-plan-metric">
                    <div class="trade-plan-metric-label">🔍 سعر الدخول:</div>
                    <div class="trade-plan-metric-value">{format_price(result['entry'])}</div>
                </div>
                <div class="trade-plan-metric">
                    <div class="trade-plan-metric-label">🎯 السعر المستهدف:</div>
                    <div class="trade-plan-metric-value">{format_price(result['target'])} <span style='font-size: 14px; color: green;'>({profit_pct:.2f}%)</span></div>
                </div>
                <div class="trade-plan-metric">
                    <div class="trade-plan-metric-label">🛑 وقف الخسارة:</div>
                    <div class="trade-plan-metric-value">{format_price(result['stop'])} <span style='font-size: 14px; color: red;'>({loss_pct:.2f}%)</span></div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        
        # Display raw metrics
        st.markdown("### 📊 المقاييس الأساسية")
        
        metrics_data = {
            "funding": {"label": "التمويل", "value": result["metrics"]["funding"], "weight": result["weights"]["funding"]},
            "oi": {"label": "OI", "value": result["metrics"]["oi"], "weight": result["weights"]["oi"]},
            "cvd": {"label": "CVD", "value": result["metrics"]["cvd"], "weight": result["weights"]["cvd"]},
            "orderbook": {"label": "دفتر الطلبات", "value": result["metrics"]["orderbook"], "weight": result["weights"]["orderbook"]},
            "backtest": {"label": "الاختبار الخلفي", "value": result["metrics"]["backtest"], "weight": result["weights"]["backtest"]}
        }

        icons = {"funding":"💰","oi":"📊","cvd":"📈","orderbook":"⚖️","backtest":"🧪"}
        
        cols = st.columns(len(metrics_data))

        for idx, k in enumerate(metrics_data):
            with cols[idx]:
                score = metrics_data[k]["value"]
                weight = metrics_data[k]["weight"]
                contrib = round(score * weight * 100, 2)
                
                st.metric(label=f"{icons[k]} {metrics_data[k]['label']}", value=f"{score:.3f}", delta=f"w={weight}")
                st.caption(f"Contribution: {contrib}%")


        st.markdown("---")
        st.markdown("### 🔍 تحليل إضافي")
        st.markdown(f"• **الدعم:** {format_price(result['raw']['support'])} | **المقاومة:** {format_price(result['raw']['resistance'])}")
        st.markdown(f"• **إشارة الشمعة:** {result['raw']['candle_signal'] if result['raw']['candle_signal'] else 'لا يوجد'}")
        
        show_raw = st.checkbox("عرض المقاييس الخام", value=False)
        if show_raw:
            st.markdown("### المقاييس الخام (من أجل الشفافية)")
            st.json(result["raw"])

    else:
        st.info("حدد الأداة/الإطار الزمني واضغط 'انطلق' للبدء.")

elif selected_page == "الحاسبة":
    trading_calculator_app()

elif selected_page == "المتتبع":
    live_market_tracker()
