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
if 'show_calculator' not in st.session_state:
    st.session_state.show_calculator = False

# Fetch all instruments once
all_instruments = fetch_instruments("SWAP") + fetch_instruments("SPOT")
if not all_instruments:
    st.error("Unable to load instruments from OKX.")
    st.stop()
    
# Title and Button in the same row
header_col1, header_col2, header_col3 = st.columns([0.6, 0.2, 0.2])
with header_col1:
    st.header("🧠 Smart Money Scanner")

def run_analysis_clicked():
    st.session_state.analysis_results = compute_confidence(st.session_state.selected_instId, st.session_state.bar)

with header_col2:
    st.markdown("""
        <style>
        div.stButton > button {
            background-image: linear-gradient(to right, #6A11CB, #2575FC);
            color: white;
            padding: 12px 30px;
            font-size: 16px;
            font-weight: bold;
            border-radius: 8px;
            border: none;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            width: 100%;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.3);
        }
        </style>
    """, unsafe_allow_html=True)
    if st.button("Go"):
        run_analysis_clicked()
        
def toggle_calculator():
    st.session_state.show_calculator = not st.session_state.show_calculator

with header_col3:
    st.markdown("""
        <style>
        div.stButton > button#calc_button {
            background-image: linear-gradient(to right, #FFA17F, #FF4B2B);
            color: white;
            padding: 12px 30px;
            font-size: 16px;
            font-weight: bold;
            border-radius: 8px;
            border: none;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            width: 100%;
        }
        div.stButton > button#calc_button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.3);
        }
        </style>
    """, unsafe_allow_html=True)
    st.button("Open Calculator", on_click=toggle_calculator, key="calc_button")
        
# Display last updated time
st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown("---")

# User inputs
st.session_state.selected_instId = st.selectbox("Select Instrument", all_instruments, index=all_instruments.index(st.session_state.selected_instId) if st.session_state.selected_instId in all_instruments else 0)
st.session_state.bar = st.selectbox("Timeframe", ["30m", "15m", "1H", "6H", "12H"], index=["30m", "15m", "1H", "6H", "12H"].index(st.session_state.bar) if st.session_state.bar in ["30m", "15m", "1H", "6H", "12H"] else 0)


# Display results if available
if st.session_state.analysis_results:
    result = st.session_state.analysis_results
    
    # Custom CSS for the cards and progress bar
    st.markdown("""
        <style>
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
    """, unsafe_allow_html=True)
    
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
        st.markdown(f"""
            <div class="trade-plan-metric">
                <div class="trade-plan-metric-label">🔍 سعر الدخول:</div>
                <div class="trade-plan-metric-value">{format_price(result['entry'])}</div>
            </div>
            <div class="trade-plan-metric">
                <div class="trade-plan-metric-label">🎯 السعر المستهدف:</div>
                <div class="trade-plan-metric-value">{format_price(result['target'])}</div>
            </div>
            <div class="trade-plan-metric">
                <div class="trade-plan-metric-label">🛑 وقف الخسارة:</div>
                <div class="trade-plan-metric-value">{format_price(result['stop'])}</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 المقاييس الأساسية")
    
    # ***هذا هو الكود المحدث لحل المشكلة***
    
    metrics_data = {
        "funding": {"label": "التمويل", "value": result["metrics"]["funding"], "weight": result["weights"]["funding"]},
        "oi": {"label": "OI", "value": result["metrics"]["oi"], "weight": result["weights"]["oi"]},
        "cvd": {"label": "CVD", "value": result["metrics"]["cvd"], "weight": result["weights"]["cvd"]},
        "orderbook": {"label": "دفتر الطلبات", "value": result["metrics"]["orderbook"], "weight": result["weights"]["orderbook"]},
        "backtest": {"label": "الاختبار الخلفي", "value": result["metrics"]["backtest"], "weight": result["weights"]["backtest"]}
    }

    icons = {"funding":"💰","oi":"📊","cvd":"📈","orderbook":"⚖️","backtest":"🧪"}
    
    # هنا تم إضافة حلقة التكرار لإنشاء الأعمدة والمقاييس داخلها
    cols = st.columns(len(metrics_data))

    for idx, k in enumerate(metrics_data):
        with cols[idx]:
            score = metrics_data[k]["value"]
            weight = metrics_data[k]["weight"]
            contrib = round(score * weight * 100, 2)
            
            # هنا يتم عرض المقياس بشكل صحيح داخل كل عمود
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

# The Trading Calculator HTML Code
calculator_html = """
<!DOCTYPE html>
<html lang="ar">
<head>
  <meta charset="UTF-8">
  <title>حاسبة التداول المتقدمة</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg-color: #0d1117;
      --card-bg: #161b22;
      --input-bg: #21262d;
      --text-color: #c9d1d9;
      --primary-color: #58a6ff;
      --success-color: #3fb950;
      --danger-color: #f85149;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: var(--bg-color);
      color: var(--text-color);
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      margin: 0;
      direction: rtl;
    }
    .card {
      background: var(--card-bg);
      padding: 30px;
      border-radius: 20px;
      width: 500px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
      border: 1px solid #30363d;
      max-width: 90%;
    }
    h2 {
      text-align: center;
      margin-bottom: 25px;
      color: var(--primary-color);
      font-weight: 600;
    }
    label {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 15px 0 5px;
      font-size: 14px;
      font-weight: 500;
    }
    .tooltip-icon {
      font-weight: bold;
      color: var(--primary-color);
      cursor: pointer;
      font-size: 16px;
      position: relative;
    }
    .tooltip-text {
      visibility: hidden;
      width: 200px;
      background-color: var(--input-bg);
      color: var(--text-color);
      text-align: center;
      border-radius: 6px;
      padding: 10px;
      position: absolute;
      z-index: 1;
      right: 0;
      top: 120%;
      opacity: 0;
      transition: opacity 0.3s;
      font-size: 12px;
      line-height: 1.5;
      border: 1px solid #444c56;
    }
    .tooltip-icon:hover .tooltip-text {
      visibility: visible;
      opacity: 1;
    }
    input, select {
      width: 100%;
      padding: 12px;
      border: 1px solid var(--input-bg);
      border-radius: 10px;
      background: var(--input-bg);
      color: var(--text-color);
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    input:focus, select:focus {
      outline: none;
      border-color: var(--primary-color);
      box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.2);
    }
    .btn-group {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 20px;
    }
    .btn-group button {
      flex: 1;
      padding: 10px;
      background: var(--input-bg);
      border: 1px solid #444c56;
      border-radius: 8px;
      color: var(--text-color);
      cursor: pointer;
      font-size: 14px;
      transition: background 0.2s, border-color 0.2s;
    }
    .btn-group button.active, .btn-group button:hover {
      background: var(--primary-color);
      border-color: var(--primary-color);
      color: #fff;
    }
    #calculateBtn {
      width: 100%;
      padding: 15px;
      background: var(--primary-color);
      border: none;
      border-radius: 12px;
      color: var(--bg-color);
      font-weight: bold;
      cursor: pointer;
      transition: background 0.2s;
    }
    #calculateBtn:hover {
      background: #478bff;
    }
    .results {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 15px;
      margin-top: 25px;
    }
    .box {
      background: var(--input-bg);
      padding: 15px;
      border-radius: 12px;
      text-align: center;
      font-weight: 500;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 70px;
    }
    .box-label {
      font-size: 12px;
      color: #8b949e;
      margin-bottom: 5px;
    }
    .box-value {
      font-size: 18px;
      font-weight: bold;
    }
    .profit { color: var(--success-color); } 
    .loss { color: var(--danger-color); } 
    .info { color: var(--primary-color); }
    canvas {
      margin-top: 25px;
      background-color: #21262d;
      border-radius: 12px;
      padding: 15px;
      border: 1px solid #30363d;
    }
    .close-btn {
        position: absolute;
        top: 10px;
        left: 10px;
        background: #f85149;
        color: white;
        border: none;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        font-size: 20px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
    }
  </style>
</head>
<body>
  <div class="card">
    <h2>📊 حاسبة التداول المتقدمة</h2>
    <label>الهامش المبدئي (IMR %)
      <span class="tooltip-icon">i<span class="tooltip-text">النسبة المئوية من إجمالي قيمة الصفقة التي تودعها.</span></span>
    </label>
    <input type="number" id="imr" placeholder="مثال: 2" oninput="updateUI()">
    <label>هامش الحِفاظ (MMR %)
      <span class="tooltip-icon">i<span class="tooltip-text">الحد الأدنى من الهامش المطلوب للحفاظ على الصفقة.</span></span>
    </label>
    <input type="number" id="mmr" placeholder="مثال: 1" oninput="updateUI()">
    <label>اختر الرافعة المالية</label>
    <div id="leverageButtons" class="btn-group"></div>
    <label>المبلغ (USDT)</label>
    <input type="number" id="capital" placeholder="مثال: 10" oninput="updateUI()">
    <label>السعر الحالي</label>
    <input type="number" id="currentPrice" placeholder="مثال: 1.00" oninput="updateUI()">
    <label>سعر الهدف</label>
    <input type="number" id="targetPrice" placeholder="مثال: 1.05" oninput="updateUI()">
    <label>الاتجاه</label>
    <select id="direction" onchange="updateUI()">
      <option value="long">📈 شراء (Long)</option>
      <option value="short">📉 بيع (Short)</option>
    </select>
    <div class="results">
      <div class="box info">
        <div class="box-label">فرق الهامش</div>
        <div class="box-value" id="marginDiff">-</div>
      </div>
      <div class="box info">
        <div class="box-label">التغير %</div>
        <div class="box-value" id="priceChangeBox">-</div>
      </div>
      <div class="box profit">
        <div class="box-label">ROI %</div>
        <div class="box-value" id="roiBox">-</div>
      </div>
      <div class="box info">
        <div class="box-label">PnL USDT</div>
        <div class="box-value" id="pnlBox">-</div>
      </div>
      <div class="box loss">
        <div class="box-label">سعر التصفية</div>
        <div class="box-value" id="liqBox">-</div>
      </div>
    </div>
    <canvas id="chart" height="120"></canvas>
  </div>
  <script>
    let chart;
    let selectedLeverage = null;
    function updateUI() {
      updateLeverageOptions();
      calculate();
    }
    function updateLeverageOptions() {
      const imr = parseFloat(document.getElementById("imr").value);
      const mmr = parseFloat(document.getElementById("mmr").value);
      const btnContainer = document.getElementById("leverageButtons");
      btnContainer.innerHTML = "";
      if (isNaN(imr) || isNaN(mmr) || imr <= mmr || imr <= 0) {
        selectedLeverage = null;
        return;
      }
      const marginDifference = imr - mmr;
      const maxLeverage = 100 / marginDifference;
      const leverageOptions = [
        { value: 5, text: `5x (منخفضة)` },
        { value: 10, text: `10x (منخفضة)` },
        { value: 20, text: `20x (متوسطة)` },
        { value: 30, text: `30x (متوسطة)` },
        { value: 50, text: `50x (عالية)` },
        { value: 100, text: `100x (عالية جداً)` }
      ];
      leverageOptions.forEach(option => {
        if (maxLeverage >= option.value) {
          const btn = document.createElement("button");
          btn.textContent = option.text;
          btn.value = option.value;
          btn.onclick = () => {
            selectedLeverage = option.value;
            document.querySelectorAll('.btn-group button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            calculate();
          };
          btnContainer.appendChild(btn);
        }
      });
      const maxBtn = document.createElement("button");
      maxBtn.textContent = `${maxLeverage.toFixed(2)}x (القصوى)`;
      maxBtn.value = maxLeverage;
      maxBtn.onclick = () => {
        selectedLeverage = maxLeverage;
        document.querySelectorAll('.btn-group button').forEach(b => b.classList.remove('active'));
        maxBtn.classList.add('active');
        calculate();
      };
      btnContainer.appendChild(maxBtn);
    }
    function calculate() {
      const imr = parseFloat(document.getElementById("imr").value);
      const mmr = parseFloat(document.getElementById("mmr").value);
      const capital = parseFloat(document.getElementById("capital").value);
      const currentPrice = parseFloat(document.getElementById("currentPrice").value);
      const targetPrice = parseFloat(document.getElementById("targetPrice").value);
      const direction = document.getElementById("direction").value;
      
      if (selectedLeverage === null || [imr, mmr, capital, currentPrice, targetPrice].some(isNaN) || imr <= mmr || imr <= 0) {
        document.getElementById("marginDiff").innerHTML = '-';
        document.getElementById("priceChangeBox").innerHTML = '-';
        document.getElementById("roiBox").innerHTML = '-';
        document.getElementById("pnlBox").innerHTML = '-';
        document.getElementById("liqBox").innerHTML = '-';
        if (chart) chart.destroy();
        return;
      }
      const leverage = selectedLeverage;
      const marginDiff = imr - mmr;
      const priceChange = ((targetPrice - currentPrice) / currentPrice) * 100;
      const actualPriceChange = direction === "short" ? -priceChange : priceChange;
      const roiPercent = actualPriceChange * leverage;
      const pnlValue = (capital * roiPercent) / 100;
      let liquidationPrice;
      if (direction === "long") {
        liquidationPrice = currentPrice * (1 - (marginDiff / 100));
      } else {
        liquidationPrice = currentPrice * (1 + (marginDiff / 100));
      }
      document.getElementById("marginDiff").innerHTML = `${marginDiff.toFixed(2)}%`;
      document.getElementById("priceChangeBox").innerHTML = `${actualPriceChange.toFixed(2)}%`;
      document.getElementById("roiBox").innerHTML = `${roiPercent.toFixed(2)}%`;
      document.getElementById("pnlBox").innerHTML = `${pnlValue.toFixed(2)} USDT`;
      document.getElementById("liqBox").innerHTML = `${liquidationPrice.toFixed(4)}`;
      let ctx = document.getElementById("chart").getContext("2d");
      if (chart) chart.destroy();
      chart = new Chart(ctx, {
        type: "line",
        data: {
          labels: ["السعر الحالي", "الهدف", "التصفية"],
          datasets: [{
            label: "السعر",
            data: [currentPrice, targetPrice, liquidationPrice],
            borderColor: "#58a6ff",
            backgroundColor: "#58a6ff",
            tension: 0.3,
            fill: false
          }]
        },
        options: {
          responsive: true,
          plugins: {
            legend: { display: false }
          },
          scales: {
            x: { 
              ticks: { color: "#c9d1d9" },
              grid: { color: "rgba(201, 209, 217, 0.1)" }
            },
            y: { 
              ticks: { color: "#c9d1d9" },
              grid: { color: "rgba(201, 209, 217, 0.1)" }
            }
          }
        }
      });
    }
    window.onload = updateUI;
  </script>
</body>
</html>
"""

# The HTML component to be rendered as a floating popup
floating_calculator_html = f"""
<div style="position: fixed; bottom: 20px; right: 20px; z-index: 9999; background: #161b22; border-radius: 20px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5); width: 500px; max-width: 90%; padding: 30px;">
    <button onclick="parent.Streamlit.setComponentValue('hide_calculator')" class="close-btn">×</button>
    {calculator_html}
</div>
"""

if st.session_state.show_calculator:
    # Use st.components.v1.html for more control and to handle JavaScript
    # Note: This is an advanced use case and requires special setup if run locally.
    # For simplicity, we'll use st.html for the floating popup.
    st.html(floating_calculator_html)
