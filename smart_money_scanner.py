import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from math import isnan
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    LOCAL_TZ = ZoneInfo("Africa/Algiers")
except Exception:
    LOCAL_TZ = None  # fallback to UTC if zoneinfo not available

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
@st.cache_data(ttl=600)
def fetch_instruments(inst_type="SWAP"):
    j = okx_get("/api/v5/public/instruments", {"instType": inst_type})
    if not j or "data" not in j:
        return []
    return [d["instId"] for d in j["data"]]

@st.cache_data(ttl=60)
def fetch_ohlcv(instId, bar="1H", limit=200):
    j = okx_get("/api/v5/market/candles", {"instId": instId, "bar": bar, "limit": str(limit)})
    if not j or "data" not in j:
        return pd.DataFrame()
    df = pd.DataFrame(j["data"], columns=["ts","o","h","l","c","vol","v2","v3","confirm"])
    for col in ["o","h","l","c","vol"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df.dropna(subset=["ts"], inplace=True)
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
    df = df.iloc[::-1].reset_index(drop=True)
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
                df = df.iloc[:, :3]
                df.columns = ["price","size","liq"]
            else:
                return pd.DataFrame()
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["size"]  = pd.to_numeric(df["size"],  errors="coerce")
            if "liq" in df.columns:
                df["liq"] = pd.to_numeric(df["liq"], errors="coerce")
            df.dropna(subset=["price","size"], inplace=True)
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
        df.rename(columns={"price":"px"}, inplace=True)
    if "sz" not in df.columns and "size" in df.columns:
        df.rename(columns={"size":"sz"}, inplace=True)
    try:
        df["px"] = pd.to_numeric(df["px"], errors="coerce")
        df["sz"] = pd.to_numeric(df["sz"], errors="coerce")
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
        df["side"] = df["side"].astype(str)
        df.dropna(subset=["px","sz","ts"], inplace=True)
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
    except Exception:
        return pd.DataFrame()
    return df.sort_values("ts").reset_index(drop=True)

# ----------------------------
# Metrics & helpers
# ----------------------------
def compute_cvd(trades_df):
    if trades_df.empty:
        return None
    signed = np.where(trades_df["side"].str.lower()=="buy", trades_df["sz"], -trades_df["sz"])
    return float(np.nansum(signed))

def orderbook_imbalance(bids, asks):
    if bids is None or asks is None or bids.empty or asks.empty:
        return None
    bid_vol = bids["size"].sum()
    ask_vol = asks["size"].sum()
    return float((bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9))

def find_liquidity_zones(bids_df, asks_df, price_range_pct=0.01):
    if bids_df is None or asks_df is None or bids_df.empty or asks_df.empty:
        return [], []
    # mid price approx from best quotes
    current_price = (bids_df["price"].max() + asks_df["price"].min()) / 2.0
    bid_zones = bids_df[bids_df["price"] > current_price * (1 - price_range_pct)]
    ask_zones = asks_df[asks_df["price"] < current_price * (1 + price_range_pct)]
    top_bids = bid_zones.nlargest(5, "size")[["price","size"]].round(6).to_dict('records')
    top_asks = ask_zones.nlargest(5, "size")[["price","size"]].round(6).to_dict('records')
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
    support = float(recent["l"].min())
    resistance = float(recent["h"].max())
    return support, resistance

def detect_candle_signal(ohlcv_df, bar):
    if ohlcv_df.empty or len(ohlcv_df) < 3:
        return None
    last = ohlcv_df.iloc[-1]
    prev = ohlcv_df.iloc[-2]
    # Engulfing
    if last["c"] > last["o"] and prev["c"] < prev["o"] and last["c"] > prev["o"] and last["o"] < prev["c"]:
        return "Bullish Engulfing"
    if last["c"] < last["o"] and prev["c"] > prev["o"] and last["c"] < prev["o"] and last["o"] > prev["c"]:
        return "Bearish Engulfing"
    # Simple Morning Star (compact)
    if bar in ["5m", "15m", "1H"] and len(ohlcv_df) >= 3:
        prev2 = ohlcv_df.iloc[-3]
        is_bearish_prev2 = prev2["c"] < prev2["o"]
        is_small_body_prev = abs(prev["o"] - prev["c"]) < ((prev["h"] - prev["l"]) * 0.2 + 1e-9)
        is_bullish_last = last["c"] > last["o"]
        if is_bearish_prev2 and is_small_body_prev and is_bullish_last:
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
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr)

# ----------------------------
# Confidence + Trade suggestion
# ----------------------------
def compute_confidence(instId, bar="1H"):
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
    atr = calculate_atr(ohlcv, period=14)

    # Normalize metrics -> 0..1
    fund_score = 0.5 + np.tanh((funding or 0.0)*500)/2 if funding is not None else 0.5
    oi_score   = 0.5 + np.tanh(np.log1p(max(oi,0.0))/20.0)/2 if oi is not None else 0.5
    cvd_score  = 0.5 + np.tanh((cvd or 0.0)/1e4)/2 if cvd is not None else 0.5
    ob_score   = ((ob_imb or 0.0)+1)/2 if ob_imb is not None else 0.5
    bt_score   = bt_win if bt_win is not None else 0.5

    metrics = {"funding": fund_score, "oi": oi_score, "cvd": cvd_score, "orderbook": ob_score, "backtest": bt_score}
    weights = {"backtest":0.30, "orderbook":0.25, "cvd":0.20, "oi":0.15, "funding":0.10}

    # clamp + weighted sum
    for k in metrics:
        v = metrics[k]
        if not isinstance(v, (float, int)) or np.isnan(v):
            metrics[k] = 0.5
        else:
            metrics[k] = float(min(1.0, max(0.0, v)))

    conf = sum(metrics[k]*weights[k] for k in metrics)
    confidence_pct = round(float(max(0.0, min(conf*100.0, 100.0))), 1)

    # Build recommendation logic
    if atr is None or price is None:
        label = "⚠️ Neutral / Mixed"
        recommendation = "Wait"
        strength = "N/A"
        entry = price
        target = stop = None
        reason = "بيانات غير كافية (ATR أو السعر مفقود)."
    else:
        bullish_strong = (confidence_pct >= 65) and (cvd or 0) > 0 and (ob_imb or 0) > 0 and (candle_signal in ["Bullish Engulfing","Bullish Morning Star"])
        bullish_weak   = (confidence_pct >= 50) and (((cvd or 0) > 0) or (candle_signal in ["Bullish Engulfing","Bullish Morning Star"]))

        bearish_strong = (confidence_pct <= 35) and (cvd or 0) < 0 and (ob_imb or 0) < 0 and (candle_signal == "Bearish Engulfing")
        bearish_weak   = (confidence_pct <= 50) and (((cvd or 0) < 0) or (candle_signal == "Bearish Engulfing"))

        if bullish_strong:
            label = "📈 Bullish"
            recommendation = "LONG"
            strength = "Strong"
            entry = price
            target = round(entry + (atr * 2.0), 6)
            stop   = round(entry - (atr * 1.0), 6)
            reason = "Confluence قوي: شمعة انعكاسية + CVD إيجابي + ميزان دفتر الأوامر لصالح المشترين."
        elif bullish_weak:
            label = "📈 Bullish"
            recommendation = "LONG"
            strength = "Weak"
            entry = price
            target = round(entry + (atr * 1.5), 6)
            stop   = round(entry - (atr * 1.0), 6)
            reason = "Confluence متوسط: إحدى الإشارات صعودية (شموع أو CVD) بينما بقية الإشارات مختلطة."
        elif bearish_strong:
            label = "📉 Bearish"
            recommendation = "SHORT"
            strength = "Strong"
            entry = price
            target = round(entry - (atr * 2.0), 6)
            stop   = round(entry + (atr * 1.0), 6)
            reason = "Confluence قوي: شمعة هبوطية + CVD سلبي + ميزان دفتر الأوامر لصالح البائعين."
        elif bearish_weak:
            label = "📉 Bearish"
            recommendation = "SHORT"
            strength = "Weak"
            entry = price
            target = round(entry - (atr * 1.5), 6)
            stop   = round(entry + (atr * 1.0), 6)
            reason = "Confluence متوسط: إحدى الإشارات هبوطية (شموع أو CVD) بينما بقية الإشارات مختلطة."
        else:
            label = "⚠️ Neutral / Mixed"
            recommendation = "Wait"
            strength = "Neutral"
            entry = price
            target = stop = None
            reason = "الإشارات متضاربة — لا يوجد سبب قوي للدخول الآن."

    # raw details
    last_bar_time_utc = None
    if not fetch_ohlcv(instId, bar, 1).empty:
        last_bar_time_utc = fetch_ohlcv(instId, bar, 1)["ts"].iloc[-1]

    raw = {
        "price": price, "funding": funding, "oi": oi, "cvd": cvd,
        "orderbook_imbalance": ob_imb, "backtest_win": bt_win,
        "support": support, "resistance": resistance,
        "candle_signal": candle_signal, "atr": atr,
        "top_bids": top_bids, "top_asks": top_asks,
        "last_bar_time_utc": str(last_bar_time_utc) if last_bar_time_utc is not None else None
    }

    return {
        "label": label, "confidence_pct": confidence_pct, "recommendation": recommendation,
        "strength": strength, "entry": entry, "target": target, "stop": stop,
        "metrics": metrics, "weights": weights, "raw": raw
    }

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Smart Money Scanner V3.8", layout="wide")
st.title("🧠 Smart Money Scanner V3.8 — Flexible Signals + Local Time")

# Time block (UTC + Local)
now_utc = datetime.now(timezone.utc)
if LOCAL_TZ:
    now_local = now_utc.astimezone(LOCAL_TZ)
    st.caption(f"⏰ Time (UTC): {now_utc.strftime('%Y-%m-%d %H:%M:%S')} | Local ({LOCAL_TZ.key}): {now_local.strftime('%Y-%m-%d %H:%M:%S')}")
else:
    st.caption(f"⏰ Time (UTC): {now_utc.strftime('%Y-%m-%d %H:%M:%S')} | Local: (UTC fallback)")

# Sidebar controls
inst_type = st.sidebar.selectbox("Instrument Type", ["SWAP", "SPOT"], index=0)
instruments = fetch_instruments(inst_type)
if not instruments:
    st.sidebar.error("Unable to load instruments from OKX.")
    st.stop()
instId = st.sidebar.selectbox("Instrument", instruments, index=0)
bar = st.sidebar.selectbox("Timeframe", ["5m","15m","1H","4H","1D"], index=2)
show_raw = st.sidebar.checkbox("Advanced: Show Raw Metrics", value=False)

# Compute
if st.sidebar.button("Compute Confidence"):
    with st.spinner("Computing — gathering live data..."):
        result = compute_confidence(instId, bar)

    # Top summary
    st.subheader(f"{result['label']} — Confidence: {result['confidence_pct']}%")
    st.metric("Live Price", f"{result['raw']['price']:,}" if result['raw']['price'] else "N/A")

    # Metrics
    st.markdown("### 📊 Core Metrics")
    icons = {"funding":"💰","oi":"📊","cvd":"📈","orderbook":"⚖️","backtest":"🧪"}
    cols = st.columns(5)
    order = ["funding","oi","cvd","orderbook","backtest"]
    for i, k in enumerate(order):
        val = result["metrics"][k]
        w = result["weights"][k]
        contrib = round(val*w*100, 2)
        with cols[i]:
            st.metric(label=f"{icons[k]} {k.upper()}", value=f"{val:.3f}", delta=f"w={w}")
            st.caption(f"Contribution: {contrib}%")

    # Trade Suggestion
    st.markdown("### 📝 Trade Suggestion")
    st.markdown(f"**Action:** {result['recommendation']} ({result['strength']})")
    st.markdown(f"**Entry:** {result['entry'] if result['entry'] is not None else 'N/A'}")
    st.markdown(f"**Target:** {result['target'] if result['target'] is not None else 'N/A'}")
    st.markdown(f"**Stop:** {result['stop'] if result['stop'] is not None else 'N/A'}")
    st.markdown(f"**Why:** { ' '.join([str(result['label']), '-', str(result['recommendation'])]) } — { ' '.join([str(result.get('strength',''))]) }")
    st.markdown(f"**Reason:** {result.get('reason','')}")

    # Structure & Candles
    st.markdown("### 🔎 Structure & Candles")
    sup = result["raw"]["support"]; res = result["raw"]["resistance"]
    st.markdown(f"• **Support:** {f'{sup:,}' if isinstance(sup,(float,int)) else 'N/A'}    \n• **Resistance:** {f'{res:,}' if isinstance(res,(float,int)) else 'N/A'}")
    st.markdown(f"• **Candle signals:** {result['raw']['candle_signal'] if result['raw']['candle_signal'] else 'None'}")

    # Liquidity snapshot (text only)
    tb = result["raw"]["top_bids"] or []
    ta = result["raw"]["top_asks"] or []
    if tb or ta:
        st.markdown("### 💧 Nearby Liquidity (text)")
        if tb:
            st.markdown("**Top Bids (near price):**")
            for r in tb:
                st.markdown(f"- price: {r.get('price')}, size: {r.get('size')}")
        if ta:
            st.markdown("**Top Asks (near price):**")
            for r in ta:
                st.markdown(f"- price: {r.get('price')}, size: {r.get('size')}")

    # Raw metrics (toggle)
    if show_raw:
        st.markdown("### Raw metrics (for transparency)")
        st.json(result["raw"])

else:
    st.info("Select instrument/timeframe and press 'Compute Confidence'.")
