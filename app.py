# app.py

import streamlit as st
import pandas as pd
from math import isnan
from datetime import datetime
from core_logic import compute_confidence, trading_calculator_app, live_market_tracker
from data_fetchers import fetch_instruments

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

# --- الدوال المساعدة (يمكن تركها هنا) ---
def format_price(price, decimals=None):
    if price is None or isnan(price):
        return "N/A"
    if decimals is None:
        if price >= 1000: decimals = 2
        elif price >= 10: decimals = 3
        else: decimals = 4
    return f"{price:,.{decimals}f}"

def calculate_pnl_percentages(entry_price, take_profit, stop_loss):
    if entry_price is None or take_profit is None or stop_loss is None or entry_price == 0:
        return None, None
    
    profit_pct = ((take_profit - entry_price) / entry_price) * 100
    loss_pct = ((stop_loss - entry_price) / entry_price) * 100
    
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
    st.markdown(f"**آخر تحديث:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("---")

st.session_state.selected_instId = st.selectbox("حدد الأداة", all_instruments, index=all_instruments.index(st.session_state.selected_instId) if st.session_state.selected_instId in all_instruments else 0)
st.session_state.bar = st.selectbox("الإطار الزمني", ["30m", "15m", "1H", "6H", "12H"], index=["30m", "15m", "1H", "6H", "12H"].index(st.session_state.bar) if st.session_state.bar in ["30m", "15m", "1H", "6H", "12H"] else 0)

# --- عرض الصفحة المحددة بناءً على شريط التنقل السفلي ---
if 'selected_leverage' not in st.session_state:
    st.session_state.selected_leverage = None

st.markdown('<div class="bottom-navbar">', unsafe_allow_html=True)
selected_page = st.radio(
    "Go to",
    ["📊 التحليل", "🧮 الحاسبة", "📈 المتتبع"],
    horizontal=True,
    label_visibility="collapsed",
    key="bottom_nav"
)
st.markdown('</div>', unsafe_allow_html=True)

if selected_page == "📊 التحليل":
    if st.session_state.analysis_results:
        result = st.session_state.analysis_results
    
        def get_confidence_color(pct):
            if pct <= 40: return "red"
            if pct <= 60: return "orange"
            return "green"

        confidence_color = get_confidence_color(result['confidence_pct'])
        progress_width = result['confidence_pct']

        rec_emoji = ""
        if result['recommendation'] == "LONG":
            rec_emoji = "🚀"
        elif result['recommendation'] == "SHORT":
            rec_emoji = "🔻"
        else:
            rec_emoji = "⏳"
        
        if result['confidence_pct'] >= 80:
            st.balloons()
            st.success("🎉 إشارة قوية جدًا تم اكتشافها! انتبه لهذه الفرصة.", icon="🔥")
        elif result['confidence_pct'] <= 20:
            st.warning("⚠️ إشارة ضعيفة جدًا. يفضل توخي الحذر.")
            
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

elif selected_page == "🧮 الحاسبة":
    trading_calculator_app()

elif selected_page == "📈 المتتبع":
    live_market_tracker()
