import streamlit as st
import pandas as pd
from math import isnan
from datetime import datetime
from core_logic import compute_confidence, trading_calculator_app, live_market_tracker
from data_fetchers import fetch_instruments

# Set wide layout and hide sidebar
st.set_page_config(layout="wide")

st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        div.st-emotion-cache-121k3k1 {
            visibility: hidden;
            display: none;
        }
        header {
            visibility: hidden;
        }
    </style>
    """, unsafe_allow_html=True)

# ... Rest of your app.py code ...
# The custom CSS for buttons, cards, etc., should be here.

# Your existing code starts from here...
# Remove the following line as it is duplicated later.
# st.markdown("<h1 style='font-size: 2.5rem; font-weight: bold; margin: 0;'>🧠 Smart Money Scanner</h1>", unsafe_allow_html=True)

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
    .custom-go-button button {
        background-image: linear-gradient(to right, #4CAF50, #2E8B57);
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 12px 30px;
        border-radius: 50px;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
        position: relative;
    }
    .custom-go-button button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        filter: brightness(1.1);
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
    
    /* --- تصميم الأزرار الجديدة --- */
    .bottom-navbar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        background-color: rgba(255, 255, 255, 0.8);
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
        padding: 10px 20px;
        display: flex;
        justify-content: space-around;
        align-items: center;
        border-top-left-radius: 20px;
        border-top-right-radius: 20px;
        backdrop-filter: blur(5px);
    }
    
    .bottom-navbar .st-cr .st-cv {
        flex-direction: row;
        justify-content: space-around;
        gap: 15px;
    }
    
    .bottom-navbar .st-cr .st-cv .st-ce label {
        display: flex;
        flex-direction: column;
        align-items: center;
        font-size: 1rem;
        font-weight: bold;
        text-align: center;
        padding: 10px 20px;
        border-radius: 50px;
        color: #6A11CB;
        background-color: #f0f0f0;
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .bottom-navbar .st-cr .st-cv .st-ce input[type="radio"] {
        display: none;
    }
    
    .bottom-navbar .st-cr .st-cv .st-ce label:hover {
        background-color: #e0e0e0;
        transform: translateY(-2px);
    }

    .bottom-navbar .st-cr .st-cv .st-ce input[type="radio"]:checked + label {
        background-image: linear-gradient(to right, #6A11CB, #2575FC);
        color: white;
        transform: translateY(-3px);
        box-shadow: 0 0 10px #6A11CB, 0 0 20px #6A11CB, 0 0 30px #2575FC; /* تأثير النيون */
    }

    /* لتحريك الأيقونة عند التحديد */
    .bottom-navbar .st-cr .st-cv .st-ce input[type="radio"]:checked + label .css-1dp5x4q {
        transform: translateY(-5px);
        transition: transform 0.3s ease;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# --- الدوال المساعدة (يمكن تركها هنا) ---
def format_price(price, decimals=None):
    if price is None or isinstance(price, (str, bool)) or isnan(price):
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
            if pct is None or isnan(pct): return "gray"
            if pct <= 40: return "red"
            if pct <= 60: return "orange"
            return "green"

        confidence_color = get_confidence_color(result['confidence_pct'])
        progress_width = result['confidence_pct'] if result['confidence_pct'] is not None and not isnan(result['confidence_pct']) else 0

        rec_emoji = ""
        if result['recommendation'] == "LONG":
            rec_emoji = "🚀"
        elif result['recommendation'] == "SHORT":
            rec_emoji = "🔻"
        else:
            rec_emoji = "⏳"
        
        if result['confidence_pct'] is not None and not isnan(result['confidence_pct']) and result['confidence_pct'] >= 80:
            st.balloons()
            st.success("🎉 إشارة قوية جدًا تم اكتشافها! انتبه لهذه الفرصة.", icon="🔥")
        elif result['confidence_pct'] is not None and not isnan(result['confidence_pct']) and result['confidence_pct'] <= 20:
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
        elif "هبوط
