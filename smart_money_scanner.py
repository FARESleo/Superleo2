import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

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
    return [d.get("instId") for d in j["data"] if "instId" in d]

@st.cache_data(ttl=45)
def fetch_ohlcv(instId, bar="1H", limit=300):
    j = okx_get("/api/v5/market/candles", {"instId": instId, "bar": bar, "limit": str(limit)})
    if not j or "data" not in j:
        return pd.DataFrame()
    cols = ["ts","o","h","l","c","vol","v2","v3","confirm"]
    try:
        df = pd.DataFrame(j["data"], columns=cols)
    except Exception:
        df = pd.DataFrame(j["data"])
        df = df.rename(columns={"open":"o","high":"h","low":"l","close":"c","volume":"vol","timestamp":"ts"})
    for col in ["o","h","l","c","vol"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(pd.to_numeric(df["ts"], errors="coerce"), unit="ms", utc=True)
    df = df.dropna(subset=["o","h","l","c"])
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
    # Normalize columns
    if "px" in df.columns:
        df["price"] = pd.to_numeric(df["px"], errors="coerce")
    if "sz" in df.columns:
        df["size"] = pd.to_numeric(df["sz"], errors="coerce")
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(pd.to_numeric(df["ts"], errors="coerce"), unit="ms", utc=True)
    if "side" in df.columns:
        df["side"] = df["side"].astype(str).str.lower()
    df = df.dropna(subset=["price","size"])
    return df.sort_values("ts").reset_index(drop=True)

# ----------------------------
# Metrics & helpers
# ----------------------------
def compute_cvd(trades_df):
    if trades_df is None or trades_df.empty:
        return None
    signed = np.where(trades_df["side"]=="buy", trades_df["size"], -trades_df["size"])
    return float(np.nansum(signed))

def orderbook_imbalance(bids, asks):
    if bids is None or asks is None or bids.empty or asks.empty:
        return None
    bid_vol = float(bids["size"].sum())
    ask_vol = float(asks["size"].sum())
    denom = bid_vol + ask_vol
    if denom <= 0:
        return None
    return float((bid_vol - ask_vol) / denom)

def simple_backtest_winrate(ohlcv_df, lookahead=6, stop_pct=0.01, rr=2.0):
    # Long-only toy backtest
    if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < 80:
        return None
    df = ohlcv_df.copy()
    df["ema20"] = df["c"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["c"].ewm(span=50, adjust=False).mean()
    delta = df["c"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100/(1+rs))
    wins, losses = 0, 0
    for i in range(50, len(df)-lookahead-1):
        row = df.iloc[i]
        cond = (row["ema20"] > row["ema50"]) and (45 <= row["rsi"] <= 70)
        if not cond:
            continue
        entry = row["c"]
        stop = entry*(1 - stop_pct)
        target = entry*(1 + stop_pct*rr)
        future = df.iloc[i+1:i+1+lookahead]
        hit_t = future["h"].ge(target).any()
        hit_s = future["l"].le(stop).any()
        if hit_t and not hit_s:
            wins += 1
        else:
            losses += 1
    total = wins + losses
    return (wins/total) if total > 0 else None

# ----------------------------
# Support/Resistance (recent swings)
# ----------------------------
def support_resistance(df, swing=10):
    if df is None or df.empty:
        return None, None
    sup = float(df["l"].rolling(swing).min().iloc[-1])
    res = float(df["h"].rolling(swing).max().iloc[-1])
    return sup, res

# ----------------------------
# Candlestick detections
# ----------------------------
def detect_candles(df):
    """Return a list of simple candle signals from the last bar vs previous."""
    signals = []
    if df is None or len(df) < 3:
        return signals
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Body/Range helpers
    def body(row): return abs(row["c"] - row["o"])
    def range_(row):
        rng = (row["h"] - row["l"])
        return rng if rng != 0 else 1e-9
    def upper_wick(row): return row["h"] - max(row["o"], row["c"])
    def lower_wick(row): return min(row["o"], row["c"]) - row["l"]

    # Engulfing
    if body(last) > body(prev) * 1.05:
        if last["c"] > last["o"] and last["o"] <= min(prev["o"], prev["c"]) and last["c"] >= max(prev["o"], prev["c"]):
            signals.append("Bullish Engulfing")
        if last["c"] < last["o"] and last["o"] >= max(prev["o"], prev["c"]) and last["c"] <= min(prev["o"], prev["c"]):
            signals.append("Bearish Engulfing")

    # Doji
    if body(last) / range_(last) < 0.1:
        signals.append("Doji")

    # Pin bars
    if (lower_wick(last) / range_(last)) > 0.6 and last["c"] > last["o"]:
        signals.append("Bullish Pin Bar")
    if (upper_wick(last) / range_(last)) > 0.6 and last["c"] < last["o"]:
        signals.append("Bearish Pin Bar")

    return signals

# ----------------------------
# Confidence engine (normalized 0..100, clamped)
# ----------------------------
def compute_confidence(instId, bar="1H"):
    ohlcv = fetch_ohlcv(instId, bar, limit=300)
    price = fetch_ticker(instId)
    funding = fetch_funding(instId)
    oi = fetch_oi(instId)
    bids, asks = fetch_orderbook(instId, depth=60)
    trades = fetch_trades(instId, limit=400)

    cvd = compute_cvd(trades) if trades is not None and not trades.empty else None
    ob_imb = orderbook_imbalance(bids, asks)
    bt_win = simple_backtest_winrate(ohlcv, lookahead=6, stop_pct=0.01, rr=2.0)

    # Normalize to 0..1 with conservative scales
    fund_score = 0.5 + np.tanh((funding or 0.0) * 500.0) / 2.0
    oi_score   = 0.5 + np.tanh(np.log1p(max(oi or 0.0, 0.0)) / 20.0) / 2.0
    cvd_score  = 0.5 + np.tanh((cvd or 0.0) / 1e4) / 2.0
    ob_score   = 0.5 if ob_imb is None else (ob_imb + 1.0) / 2.0
    bt_score   = 0.5 if bt_win is None else float(np.clip(bt_win, 0.0, 1.0))

    metrics = {
        "funding": float(np.clip(fund_score, 0.0, 1.0)),
        "oi"     : float(np.clip(oi_score,     0.0, 1.0)),
        "cvd"    : float(np.clip(cvd_score,    0.0, 1.0)),
        "orderbook": float(np.clip(ob_score,   0.0, 1.0)),
        "backtest" : float(np.clip(bt_score,   0.0, 1.0)),
    }

    weights = {
        "backtest": 0.30,
        "orderbook": 0.25,
        "cvd": 0.20,
        "oi": 0.15,
        "funding": 0.10,
    }

    conf = sum(metrics[k] * weights[k] for k in metrics.keys())
    confidence_pct = float(np.clip(conf * 100.0, 0.0, 100.0))
    confidence_pct = round(confidence_pct, 1)

    if confidence_pct >= 65:
        label = "📈 Bullish"
    elif confidence_pct <= 35:
        label = "📉 Bearish"
    else:
        label = "⚠️ Neutral / Mixed"

    # Candlestick + S/R
    sup, res = support_resistance(ohlcv, swing=10)
    candles = detect_candles(ohlcv)

    raw = {
        "price": price,
        "funding": funding,
        "oi": oi,
        "cvd": cvd,
        "orderbook_imbalance": ob_imb,
        "backtest_win": bt_win,
        "support": sup,
        "resistance": res,
        "candles": candles,
    }

    return {
        "label": label,
        "confidence_pct": confidence_pct,
        "metrics": metrics,
        "weights": weights,
        "raw": raw
    }

# ----------------------------
# Trade suggestion (Entry/Target/Stop + Reasoning)
# ----------------------------
def suggest_trade(conf_label, price, support, resistance, ob_imb, candles):
    action = "Neutral / Wait"
    entry = price
    target = None
    stop = None
    rationale = []

    if candles:
        rationale.append("Candles: " + ", ".join(candles))

    if ob_imb is not None:
        if ob_imb > 0.10:
            rationale.append("Orderbook shows bid dominance (buy-side liquidity).")
        elif ob_imb < -0.10:
            rationale.append("Orderbook shows ask dominance (sell-side liquidity).")
        else:
            rationale.append("Orderbook balanced.")

    if conf_label == "📈 Bullish":
        action = "LONG (Buy)"
        target = (resistance * 0.995) if resistance else (price * 1.02 if price else None)
        stop   = (support * 0.995)    if support    else (price * 0.99 if price else None)
        rationale.append("Bias: Bullish (confidence engine).")
        if support and resistance:
            rationale.append(f"Trade within S/R range: support≈{support:.4f}, resistance≈{resistance:.4f}.")
    elif conf_label == "📉 Bearish":
        action = "SHORT (Sell)"
        target = (support * 1.005)    if support    else (price * 0.98 if price else None)
        stop   = (resistance * 1.005) if resistance else (price * 1.01 if price else None)
        rationale.append("Bias: Bearish (confidence engine).")
        if support and resistance:
            rationale.append(f"Trade within S/R range: support≈{support:.4f}, resistance≈{resistance:.4f}.")
    else:
        rationale.append("Bias: Neutral — wait for clearer confluence.")

    def safe(v):
        return v if (v is not None and np.isfinite(v) and v > 0) else None

    entry  = safe(entry)
    target = safe(target)
    stop   = safe(stop)

    rr = None
    if entry and target and stop:
        if action.startswith("LONG"):
            rr = (target - entry) / max(1e-9, entry - stop)
        elif action.startswith("SHORT"):
            rr = (entry - target) / max(1e-9, stop - entry)
        if rr is not None and rr < 1.0:
            rationale.append("Warning: Risk/Reward < 1.0, consider skipping or refining levels.")

    return {
        "action": action,
        "entry": entry,
        "target": target,
        "stop": stop,
        "rr": rr,
        "reason": " ".join(rationale)
    }

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Smart Money Scanner V3.7", layout="wide")
st.title("🧠 Smart Money Scanner V3.7 — Integrated Signals + Trade Suggestion")

inst_type = st.sidebar.selectbox("Instrument Type", ["SWAP", "SPOT"])
instruments = fetch_instruments(inst_type)
if not instruments:
    st.sidebar.error("Unable to load instruments from OKX. Try again later.")
    st.stop()

instId = st.sidebar.selectbox("Instrument", instruments, index=0)
bar = st.sidebar.selectbox("Timeframe", ["1m","5m","15m","1H","4H","1D"], index=3)
show_raw = st.sidebar.checkbox("Advanced: Show Raw Metrics", value=False)

if st.sidebar.button("Compute Confidence"):
    with st.spinner("Gathering live data and computing confidence..."):
        res = compute_confidence(instId, bar)

    st.subheader(f"{res['label']} — Confidence: {res['confidence_pct']}%")
    live_price = res["raw"]["price"]
    st.metric("Live Price", f"{live_price:,.6f}" if live_price is not None else "N/A")

    icons = {"funding":"💰","oi":"📊","cvd":"📈","orderbook":"⚖️","backtest":"🧪"}
    cols = st.columns(5)
    order = ["funding","oi","cvd","orderbook","backtest"]
    for i, k in enumerate(order):
        col = cols[i]
        score = res["metrics"].get(k, 0.5)
        w = res["weights"].get(k, 0.0)
        contrib = round(score * w * 100, 2)
        col.metric(label=f"{icons[k]} {k.upper()}", value=f"{score:.3f}", delta=f"w={w}")
        col.caption(f"Contribution: {contrib}%")

    sup = res["raw"]["support"]; resis = res["raw"]["resistance"]; candles = res["raw"]["candles"]
    st.markdown("### 🔎 Structure & Candles")
    st.write(f"• Support (approx): **{sup:.6f}**" if sup else "• Support: N/A")
    st.write(f"• Resistance (approx): **{resis:.6f}**" if resis else "• Resistance: N/A")
    st.write("• Candle signals: " + (", ".join(candles) if candles else "None"))

    idea = suggest_trade(
        conf_label=res["label"],
        price=live_price,
        support=sup,
        resistance=resis,
        ob_imb=res["raw"]["orderbook_imbalance"],
        candles=candles,
    )

    st.markdown("### 📝 Trade Suggestion")
    st.write(f"**Action**: {idea['action']}")
    st.write(f"**Entry**: {idea['entry']:.6f}" if idea['entry'] else "**Entry**: N/A")
    st.write(f"**Target**: {idea['target']:.6f}" if idea['target'] else "**Target**: N/A")
    st.write(f"**Stop**: {idea['stop']:.6f}" if idea['stop'] else "**Stop**: N/A")
    if idea["rr"] is not None and np.isfinite(idea["rr"]):
        st.write(f"**R:R** ≈ {idea['rr']:.2f}")
    st.write(f"**Why**: {idea['reason']}")

    if show_raw:
        st.markdown("### Raw metrics (for transparency)")
        st.json(res["raw"])
else:
    st.info("Select instrument/timeframe and press 'Compute Confidence' in the sidebar.")
