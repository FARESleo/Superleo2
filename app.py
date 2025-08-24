import streamlit as st
import pandas as pd
from math import isnan
from datetime import datetime
from core_logic import compute_confidence, trading_calculator_app, live_market_tracker
from data_fetchers import fetch_instruments

# --- كود CSS المعدل لتصميم الواجهة ---
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
    
    /* --- تعديل زر Go: شكل أكثر دائرية، حجم أكبر، ألوان أزرق --- */
    .custom-go-button button {
        background-image: linear-gradient(to right, #2196F3, #0D47A1);
        color: white;
        font-size: 1.3rem;
        font-weight: bold;
        padding: 15px 35px;
        border-radius: 100px;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
        position: relative;
    }
    .custom-go-button button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        filter: brightness(1.1);
    }

    /* --- الكاردز العامة: تغيير إلى خلفية بيضاء مع حواف أزرق --- */
    .custom-card {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(33, 150, 243, 0.2);
        color: #333;
    }
    .card-header {
        font-size: 14px;
        color: #555;
        text-transform: uppercase;
        font-weight: bold;
    }
    .card-value {
        font-size: 30px;
        font-weight: bold;
        margin-top: 5px;
    }
    .progress-bar-container {
        background-color: #e0e0e0;
        border-radius: 50px;
        height: 12px;
        width: 100%;
        margin-top: 10px;
    }
    .progress-bar {
        height: 100%;
        border-radius: 50px;
        transition: width 0.5s ease-in-out;
    }
    
    /* --- كارد خطة التداول: تغيير حواف إلى أزرق --- */
    .trade-plan-card {
        background-color: #f0f4ff;
        border-left: 5px solid #2196F3;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 20px;
    }
    .trade-plan-title {
        font-size: 26px;
        font-weight: bold;
        color: #0D47A1;
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
    
    /* --- كارد الأسباب: تعديل الألوان والشكل --- */
    .reason-card {
        background-color: #f0f4f7;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        border-left: 5px solid;
    }
    .reason-text {
        font-size: 16px;
        line-height: 1.6;
        margin-top: 5px;
        font-style: italic;
    }
    .reason-card.bullish {
        border-color: #4CAF50;
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .reason-card.bearish {
        border-color: #d32f2f;
        background-color: #ffebee;
        color: #b71c1c;
    }
    .reason-card.neutral {
        border-color: #ff9800;
        background-color: #fff3e0;
        color: #e65100;
    }
    
    /* --- تعديل الشريط السفلي: ألوان أزرق-أخضر، شكل أكثر انسيابية --- */
    .bottom-navbar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        background-color: rgba(255, 255, 255, 0.9);
        box-shadow: 0 -2px 15px rgba(0, 0, 0, 0.15);
        padding: 12px 25px;
        display: flex;
        justify-content: space-around;
        align-items: center;
        border-top-left-radius: 25px;
        border-top-right-radius: 25px;
        backdrop-filter: blur(8px);
    }
    
    .bottom-navbar .st-cr .st-cv {
        flex-direction: row;
        justify-content: space-around;
        gap: 20px;
    }
    
    .bottom-navbar .st-cr .st-cv .st-ce label {
        display: flex;
        flex-direction: column;
        align-items: center;
        font-size: 1.1rem;
        font-weight: bold;
        text-align: center;
        padding: 12px 25px;
        border-radius: 100px;
        color: #2196F3;
        background-color: #e3f2fd;
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .bottom-navbar .st-cr .st-cv .st-ce input[type="radio"] {
        display: none;
    }
    
    .bottom-navbar .st-cr .st-cv .st-ce label:hover {
        background-color: #bbdefb;
        transform: translateY(-3px);
    }

    .bottom-navbar .st-cr .st-cv .st-ce input[type="radio"]:checked + label {
        background-image: linear-gradient(to right, #2196F3, #4CAF50);
        color: white;
        transform: translateY(-4px);
        box-shadow: 0 0 12px #2196F3, 0 0 24px #2196F3, 0 0 36px #4CAF50;
    }

    .bottom-navbar .st-cr .st-cv .st-ce input[type="radio"]:checked + label .css-1dp5x4q {
        transform: translateY(-5px);
        transition: transform 0.3s ease;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# --- الدوال المساعدة ---
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
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "📊 التحليل"  # تعريف مبدئي للصفحة

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
    with st.container(border=False):
        st.markdown('<div class="custom-go-button">', unsafe_allow_html=True)
        if st.button("🚀 ابدأ التحليل!", use_container_width=True):
            run_analysis_clicked()
        st.markdown('</div>', unsafe_allow_html=True)

with header_col2:
    st.markdown(f"**آخر تحديث:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("---")

st.session_state.selected_instId = st.selectbox("حدد الأداة", all_instruments, index=all_instruments.index(st.session_state.selected_instId) if st.session_state.selected_instId in all_instruments else 0)
st.session_state.bar = st.selectbox("الإطار الزمني", ["30m", "15m", "1H", "6H", "12H"], index=["30m", "15m", "1H", "6H", "12H"].index(st.session_state.bar) if st.session_state.bar in ["30m", "15m", "1H", "6H", "12H"] else 0)

# --- تعريف الشريط السفلي للتنقل ---
st.markdown('<div class="bottom-navbar">', unsafe_allow_html=True)
st.session_state.selected_page = st.radio(
    "Go to",
    ["📊 التحليل", "🧮 الحاسبة", "📈 المتتبع"],
    horizontal=True,
    label_visibility="collapsed",
    key="bottom_nav"
)
st.markdown('</div>', unsafe_allow_html=True)

# --- عرض الصفحة المحددة بناءً على الشريط السفلي ---
if st.session_state.selected_page == "📊 التحليل":
    st.markdown('<div style="background-color: rgba(227, 242, 253, 0.8); padding: 20px; border-radius: 15px;">', unsafe_allow_html=True)
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
        st.info("حدد الأداة/الإطار الزمني واضغط 'ابدأ التحليل!' للبدء.")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.selected_page == "🧮 الحاسبة":
    st.markdown('<div style="background-color: rgba(232, 245, 233, 0.8); padding: 20px; border-radius: 15px;">', unsafe_allow_html=True)
    trading_calculator_app()
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.selected_page == "📈 المتتبع":
    st.markdown('<div style="background-color: rgba(255, 243, 224, 0.8); padding: 20px; border-radius: 15px;">', unsafe_allow_html=True)
    live_market_tracker()
    st.markdown('</div>', unsafe_allow_html=True)
