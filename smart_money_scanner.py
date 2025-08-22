import streamlit as st
import requests, time
import pandas as pd
import numpy as np
from math import isnan

OKX_BASE = "https://www.okx.com"

# -------------------------
# HTTP helper with retries
# -------------------------
def okx_get(path, params=None, retries=3, delay=0.7):
    url = f"{OKX_BASE}{path}"
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(delay*(i+1))
    return None

# -------------------------
# Data fetchers (cached)
# -------------------------
@st.cache_data(ttl=60)
def fetch_ohlcv(instId, bar="1H", limit=200):
    j = okx_get("/api/v5/market/candles", {"instId": instId, "bar": bar, "limit": str(limit)})
    if not j or "data" not in j:
        return pd.DataFrame()
    df = pd.DataFrame(j["data"], columns=["ts","o","h","l","c","vol","v2","v3","confirm"])
    for col in ["o","h","l","c","vol"]:
        df[col] = df[col].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
    return df.iloc[::-1].reset_index(drop=True)

@st.cache_data(ttl=30)
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
    j = okx_get("/api/v5/market/books", {"instId": instId, "sz": depth})
    if not j or "data" not in j:
        return None, None
    ob = j["data"][0]
    bids = pd.DataFrame(ob.get("bids",[]), columns=["price","size","liq"]).astype(float)
    asks = pd.DataFrame(ob.get("asks",[]), columns=["price","size","liq"]).astype(float)
    return bids, asks

@st.cache_data(ttl=30)
def fetch_trades(instId, limit=400):
    j = okx_get("/api/v5/market/trades", {"instId": instId, "limit": str(limit)})
    if not j or "data" not in j:
        return pd.DataFrame()
    df = pd.DataFrame(j["data"])
    # expected columns: instId, tradeId, px, sz, side, ts
    df = df.rename(columns={0:"inst"}) if 0 in df.columns else df
    # ensure proper columns
    if "px" not in df.columns:
        df = df.rename(columns={ "price":"px", "size":"sz"}) if "price" in df.columns else df
    try:
        df["px"] = df["px"].astype(float)
        df["sz"] = df["sz"].astype(float)
        df["side"] = df["side"].astype(str)
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
    except Exception:
        pass
    return df.sort_values("ts").reset_index(drop=True)

# -------------------------
# Simple indicators & norm
# -------------------------
def compute_cvd(trades_df):
    if trades_df.empty:
        return None
    trades_df = trades_df.copy()
    trades_df["signed"] = np.where(trades_df["side"].str.lower()=="buy", trades_df["sz"], -trades_df["sz"])
    return trades_df["signed"].cumsum().iloc[-1]

def orderbook_imbalance_score(bids, asks):
    if bids is None or asks is None or bids.empty or asks.empty:
        return None
    bid_vol = bids["size"].sum()
    ask_vol = asks["size"].sum()
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)  # -1 .. +1
    return float(imbalance)

def z_to_01(z):  # map z-score to 0..1 via logistic-ish
    # keep stable numeric transform
    return 1.0 / (1.0 + np.exp(-z/1.0))

def percentile_to_score(val, hist_vals):
    # safe mapping using empirical CDF
    if hist_vals is None or len(hist_vals)==0 or val is None or isnan(val):
        return None
    arr = np.array(hist_vals)
    # avoid NaNs
    arr = arr[~np.isnan(arr)]
    if len(arr)==0:
        return None
    p = (arr < val).sum() / len(arr)
    return float(p)  # 0..1

# -------------------------
# Backtest (simple) used for calibration and a feature
# -------------------------
def simple_backtest_winrate(df_ohlcv, lookahead=6, stop_pct=0.01, rr=2.0):
    # conservative long-only rule: EMA20>EMA50 & RSI in [45,70]
    if df_ohlcv.empty or len(df_ohlcv)<60:
        return None
    df = df_ohlcv.copy()
    df["ema20"] = df["c"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["c"].ewm(span=50, adjust=False).mean()
    delta = df["c"].diff()
    gain = (delta.clip(lower=0)).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100/(1+rs))
    wins=losses=0
    for i in range(50, len(df)-lookahead-1):
        row = df.iloc[i]
        cond = (row["ema20"]>row["ema50"]) and (45<=row["rsi"]<=70)
        if not cond:
            continue
        entry = row["c"]
        stop = entry*(1-stop_pct)
        target = entry*(1+stop_pct*rr)
        future = df.iloc[i+1:i+1+lookahead]
        hit_t = future["h"].ge(target).any()
        hit_s = future["l"].le(stop).any()
        if hit_t and not hit_s:
            wins+=1
        else:
            losses+=1
    total = wins+losses
    return (wins/total) if total>0 else None

# -------------------------
# Confidence engine: gather series history & map to 0..1 scores
# -------------------------
def build_historical_series(instId, bar="1H"):
    # gather historical snapshots for normalization (we use OHLCV windows)
    df = fetch_ohlcv(instId, bar, limit=300)
    # funding history: call several times? OKX doesn't provide long history easily via this API; we use short window of recent funding calls repeated (practical compromise)
    # For simplicity: we derive history samples from sliding windows:
    hist = {}
    if not df.empty:
        # orderbook snapshots not stored historically here; we'll approximate using returns/volatility
        hist["volatility"] = df["c"].pct_change().rolling(14).std().dropna().tolist()
        hist["close_changes"] = df["c"].pct_change().dropna().tolist()
    return hist

def compute_confidence(instId, bar="1H"):
    # 1. fetch current values
    ohlcv = fetch_ohlcv(instId, bar, limit=250)
    price = fetch_ticker(instId)
    funding = fetch_funding(instId)
    oi = fetch_oi(instId)
    bids, asks = fetch_orderbook(instId, depth=60)
    trades = fetch_trades(instId, limit=400)

    # 2. derived metrics
    cvd = compute_cvd(trades)
    ob_imb = orderbook_imbalance_score(bids, asks)
    backtest_win = simple_backtest_winrate(ohlcv, lookahead=6, stop_pct=0.01, rr=2.0)

    # 3. build lightweight historical series for percentile mapping (local method)
    hist = build_historical_series(instId, bar)
    # map values to 0..1 scores (directional: higher -> bullish)
    # funding: positive -> longs pay shorts -> bullish pressure (map via simple percentile using small synthetic series)
    # Since funding history via API is limited, we'll use simple sign and magnitude heuristics
    fund_score = None
    if funding is not None:
        # assume funding typical range +/-0.002 (0.2%), scale
        fund_score = 0.5 + np.tanh(funding*500)/2  # bias to 0..1, centered at 0.5

    oi_score = None
    if oi is not None and not ohlcv.empty:
        # compute OI change over last two candles if possible using a fresh call for previous OI (approximate by calling API twice is overkill)
        # simpler: compare current oi to rolling mean of recent oks via small sample if available
        oi_score = 0.5 + np.tanh((oi / (1e6+abs(oi))) )/2  # coarse mapping; will refine in calibration

    cvd_score = None
    if cvd is not None:
        # cvd positive -> buy pressure -> convert via percentile using recent trade deltas
        if not trades.empty:
            recent_deltas = trades["sz"].where(trades["side"]=="buy", -trades["sz"]).cumsum().diff().dropna()
            cvd_score = percentile_to_score(cvd, recent_deltas.tolist()) if len(recent_deltas)>10 else (0.5 + np.tanh(cvd/1e4)/2)
        else:
            cvd_score = 0.5

    ob_score = None
    if ob_imb is not None:
        # imbalance [-1..1] -> map to 0..1 (positive = more bids)
        ob_score = (ob_imb + 1.0) / 2.0

    backtest_score = None
    if backtest_win is not None:
        # backtest returns 0..1 -> map directly
        backtest_score = backtest_win

    # 4. combine with weights (initial weights; will calibrate later)
    weights = {
        "backtest": 0.30,
        "orderbook": 0.25,
        "cvd": 0.20,
        "oi": 0.15,
        "funding": 0.10
    }
    # collect scores (fill missing with neutral 0.5)
    scores = {
        "funding": fund_score if fund_score is not None else 0.5,
        "oi": oi_score if oi_score is not None else 0.5,
        "cvd": cvd_score if cvd_score is not None else 0.5,
        "orderbook": ob_score if ob_score is not None else 0.5,
        "backtest": backtest_score if backtest_score is not None else 0.5
    }
    # ensure 0..1 bounds
    for k in scores:
        v = scores[k]
        try:
            if v is None or isnan(v):
                scores[k] = 0.5
            else:
                scores[k] = min(1.0, max(0.0, float(v)))
        except:
            scores[k] = 0.5

    # weighted sum -> confidence 0..1
    conf = 0.0
    for k,w in weights.items():
        conf += scores.get(k,0.5) * w
    # normalize by weights sum (should be 1)
    conf_pct = round(conf * 100, 1)

    # derive final label
    label = "⚠️ Neutral / Mixed"
    if conf_pct >= 65:
        label = "📈 Bullish"
    elif conf_pct <= 35:
        label = "📉 Bearish"
    else:
        label = "⚠️ Neutral / Mixed"

    # return structured result with explainability
    return {
        "price": price,
        "scores": scores,
        "weights": weights,
        "confidence_pct": conf_pct,
        "label": label,
        "raw_metrics": {
            "funding": funding,
            "oi": oi,
            "cvd": cvd,
            "orderbook_imbalance": ob_imb,
            "backtest_win": backtest_win
        }
    }

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Smart Money Scanner V3.4", layout="wide")
st.title("🧠 Smart Money Scanner V3.4 — Confidence Engine (OKX)")

# instrument selection via OKX instruments endpoint
inst_type = st.sidebar.selectbox("Instrument Type", ["SWAP","SPOT"])
# load instrument list lazily
@st.cache_data
def load_instruments(inst_type):
    jr = okx_get("/api/v5/public/instruments", {"instType": inst_type})
    if not jr or "data" not in jr:
        return []
    return [d["instId"] for d in jr["data"]]

instruments = load_instruments(inst_type)
if not instruments:
    st.sidebar.error("Unable to load instruments from OKX. Check network.")
    st.stop()

instId = st.sidebar.selectbox("Instrument", instruments, index=0)
bar = st.sidebar.selectbox("Timeframe", ["1m","5m","15m","1H","4H","1D"], index=3)
if st.sidebar.button("Compute Confidence"):
    with st.spinner("Gathering data and computing confidence..."):
        res = compute_confidence(instId, bar)
    # display
    st.subheader(f"{res['label']} — Confidence: {res['confidence_pct']}%")
    st.metric("Live Price", res["price"] if res["price"] is not None else "N/A")
    st.markdown("### Breakdown (score 0..1) and weights")
    cols = st.columns(5)
    i=0
    for k in ["backtest","orderbook","cvd","oi","funding"]:
        col = cols[i]
        score = round(res["scores"][k],3)
        w = res["weights"][k]
        contribution = round(score * w * 100,2)
        col.metric(k.upper(), f"{score}", delta=f"weight {w}")
        col.caption(f"contrib: {contribution}%")
        i+=1
    st.markdown("### Raw metrics (for debugging / transparency)")
    st.json(res["raw_metrics"])
    st.caption("ملاحظة: هذه نسبة ثقة مبدئية مبنية على مزيج من الإشارات. لتحسينها: قم بعملية معايرة (Calibration) عبر Backtest أطول ودمج مصادر إضافية (WebSocket, multiple exchanges).")

else:
    st.info("اختر Instrument ثم اضغط Compute Confidence")

# Footer quick tips
st.markdown("---")
st.caption("V3.4 — Confidence Engine. استخدمها كإشارة مساعدة، لا تعتمد عليها وحدها. للمستخدم المتقدم: أنصح بتمكين WebSocket وتوسيع Backtest عبر سنوات بيانات لتحسين الأوزان تلقائيًا.")
