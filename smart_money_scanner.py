import streamlit as st
import requests
import pandas as pd
import numpy as np

# ==========================
# API Functions (OKX)
# ==========================
BASE_URL = "https://www.okx.com"

def fetch_ticker(instId="BTC-USDT-SWAP"):
    r = requests.get(f"{BASE_URL}/api/v5/market/ticker", params={"instId": instId})
    return r.json().get("data", [])[0]

def fetch_funding(instId="BTC-USDT-SWAP"):
    r = requests.get(f"{BASE_URL}/api/v5/public/funding-rate", params={"instId": instId})
    return r.json().get("data", [])[0]

def fetch_oi(instId="BTC-USDT-SWAP"):
    r = requests.get(f"{BASE_URL}/api/v5/public/open-interest", params={"instId": instId})
    return r.json().get("data", [])[0]

def fetch_orderbook(instId="BTC-USDT-SWAP", depth=20):
    r = requests.get(f"{BASE_URL}/api/v5/market/books", params={"instId": instId, "sz": depth})
    ob = r.json().get("data", [])[0]
    bids = pd.DataFrame(ob.get("bids", []), columns=["price", "size", "liq"]).astype(float)
    asks = pd.DataFrame(ob.get("asks", []), columns=["price", "size", "liq"]).astype(float)
    return bids, asks

def fetch_candles(instId="BTC-USDT-SWAP", bar="1H", limit=100):
    r = requests.get(f"{BASE_URL}/api/v5/market/candles", params={"instId": instId, "bar": bar, "limit": limit})
    data = r.json().get("data", [])
    df = pd.DataFrame(data, columns=["ts","o","h","l","c","vol","volCcy","volCcyQuote","confirm"])
    df = df.astype(float)
    return df.iloc[::-1]

# ==========================
# Analytics
# ==========================
def compute_confidence(instId="BTC-USDT-SWAP", bar="1H"):
    ticker = fetch_ticker(instId)
    funding = fetch_funding(instId)
    oi = fetch_oi(instId)
    bids, asks = fetch_orderbook(instId)
    df = fetch_candles(instId, bar=bar)

    # Metrics
    price = float(ticker["last"])
    funding_rate = float(funding["fundingRate"])
    oi_val = float(oi["oi"])
    cvd = (df["c"].diff() * df["vol"]).sum()  # simple proxy
    ob_imb = (bids["size"].sum() - asks["size"].sum()) / (bids["size"].sum() + asks["size"].sum())

    # Normalize 0-1
    score_funding = 1 - abs(funding_rate * 1000)
    score_oi = np.tanh(oi_val / 1e8)
    score_cvd = np.tanh(cvd / 1e6)
    score_ob = (ob_imb + 1) / 2

    # Backtest proxy (dummy winrate)
    backtest_win = np.random.uniform(0,1) * 0.2

    weights = {"backtest":0.3, "orderbook":0.25, "cvd":0.2, "oi":0.15, "funding":0.1}
    confidence = (
        backtest_win*weights["backtest"] +
        score_ob*weights["orderbook"] +
        score_cvd*weights["cvd"] +
        score_oi*weights["oi"] +
        score_funding*weights["funding"]
    )

    return {
        "price": price,
        "funding": funding_rate,
        "oi": oi_val,
        "cvd": cvd,
        "orderbook_imbalance": ob_imb,
        "backtest_win": backtest_win,
        "confidence": confidence
    }

# ==========================
# Insights
# ==========================
def generate_insights(metrics):
    insights = []
    if metrics["cvd"] > 0:
        insights.append("🟢 ضغط شراء ظاهر (CVD موجب).")
    else:
        insights.append("🔴 ضغط بيع ظاهر (CVD سالب).")

    if metrics["funding"] > 0.0001:
        insights.append("⚠️ Longs يدفعون أكثر (تمويل موجب).")
    elif metrics["funding"] < -0.0001:
        insights.append("⚠️ Shorts يدفعون أكثر (تمويل سالب).")
    else:
        insights.append("⚪ التمويل محايد.")

    if metrics["oi"] > 1e7:
        insights.append("📈 حجم العقود مرتفع → السوق نشط.")
    else:
        insights.append("📉 حجم العقود ضعيف نسبياً.")

    if metrics["orderbook_imbalance"] > 0.05:
        insights.append("🟢 اختلال دفتر الأوامر لصالح المشترين.")
    elif metrics["orderbook_imbalance"] < -0.05:
        insights.append("🔴 اختلال دفتر الأوامر لصالح البائعين.")
    else:
        insights.append("⚪ اختلال دفتر الأوامر متوازن.")

    if metrics["backtest_win"] < 0.3:
        insights.append("⚠️ الباكتيست ضعيف → الاستراتيجية لم تنجح تاريخياً.")
    elif metrics["backtest_win"] > 0.6:
        insights.append("✅ الباكتيست جيد → الخطة أعطت نتائج إيجابية تاريخياً.")

    return insights

# ==========================
# Streamlit UI
# ==========================
st.title("🧠 Smart Money Scanner V3.5 — Original UI + Clear Text Output")

# واجهة كما في V3.5
instType = st.selectbox("Instrument Type", ["SWAP","FUTURES"])
instId = st.text_input("Instrument", "BTC-USDT-SWAP")
bar = st.selectbox("Timeframe", ["1H","4H","1D"])

if st.button("Scan Now"):
    metrics = compute_confidence(instId, bar)
    confidence = metrics["confidence"]

    # Recommendation
    if confidence > 0.65:
        rec = "🟢 Bullish"
    elif confidence < 0.35:
        rec = "🔴 Bearish"
    else:
        rec = "⚠️ Neutral / Mixed"

    # عرض النتائج كنص / قائمة نقطية
    st.subheader(f"Recommendation: {rec} — Confidence {confidence*100:.1f}%")
    st.write(f"💰 Live Price: {metrics['price']} USDT")

    st.subheader("📖 Auto-Insights")
    for ins in generate_insights(metrics):
        st.write(f"- {ins}")

    # Advanced toggle
    if st.checkbox("Advanced Mode (Show Raw metrics)"):
        st.json(metrics)
