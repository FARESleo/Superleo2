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
        background-image: linear-gradient(to right, #2196F3, #0D47A1); /* تغيير إلى تدرج أزرق */
        color: white;
        font-size: 1.3rem; /* زيادة الحجم */
        font-weight: bold;
        padding: 15px 35px; /* زيادة البادينج */
        border-radius: 100px; /* أكثر دائرية */
        border: none;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
        position: relative;
    }
    .custom-go-button button:hover {
        transform: scale(1.05); /* تأثير تكبير بدلاً من رفع */
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        filter: brightness(1.1);
    }

    /* --- الكاردز العامة: تغيير إلى خلفية بيضاء مع حواف أزرق --- */
    .custom-card {
        background-color: #FFFFFF; /* أبيض بدلاً من رمادي فاتح */
        border-radius: 15px; /* حواف أكثر انحناء */
        padding: 25px; /* زيادة البادينج */
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(33, 150, 243, 0.2); /* ظل أزرق خفيف */
        color: #333;
    }
    .card-header {
        font-size: 14px;
        color: #555; /* تغيير إلى رمادي أغمق */
        text-transform: uppercase;
        font-weight: bold;
    }
    .card-value {
        font-size: 30px; /* زيادة الحجم */
        font-weight: bold;
        margin-top: 5px;
    }
    .progress-bar-container {
        background-color: #e0e0e0; /* رمادي أفتح */
        border-radius: 50px;
        height: 12px; /* سمك أكبر */
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
        background-color: #f0f4ff; /* خلفية أزرق فاتح */
        border-left: 5px solid #2196F3; /* حواف أزرق */
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 20px;
    }
    .trade-plan-title {
        font-size: 26px;
        font-weight: bold;
        color: #0D47A1; /* أزرق غامق */
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
        border-radius: 12px; /* حواف أكثر */
        padding: 20px; /* زيادة */
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
        background-color: #e8f5e9; /* أخضر فاتح أكثر */
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
        background-color: rgba(255, 255, 255, 0.9); /* شفافية أعلى */
        box-shadow: 0 -2px 15px rgba(0, 0, 0, 0.15);
        padding: 12px 25px;
        display: flex;
        justify-content: space-around;
        align-items: center;
        border-top-left-radius: 25px;
        border-top-right-radius: 25px;
        backdrop-filter: blur(8px); /* بلور أقوى */
    }
    
    .bottom-navbar .st-cr .st-cv {
        flex-direction: row;
        justify-content: space-around;
        gap: 20px; /* مسافة أكبر */
    }
    
    .bottom-navbar .st-cr .st-cv .st-ce label {
        display: flex;
        flex-direction: column;
        align-items: center;
        font-size: 1.1rem; /* حجم أكبر */
        font-weight: bold;
        text-align: center;
        padding: 12px 25px;
        border-radius: 100px; /* أكثر دائرية */
        color: #2196F3; /* أزرق */
        background-color: #e3f2fd; /* خلفية أزرق فاتح */
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .bottom-navbar .st-cr .st-cv .st-ce input[type="radio"] {
        display: none;
    }
    
    .bottom-navbar .st-cr .st-cv .st-ce label:hover {
        background-color: #bbdefb; /* hover أزرق أغمق */
        transform: translateY(-3px); /* رفع أقوى */
    }

    .bottom-navbar .st-cr .st-cv .st-ce input[type="radio"]:checked + label {
        background-image: linear-gradient(to right, #2196F3, #4CAF50); /* تدرج أزرق-أخضر */
        color: white;
        transform: translateY(-4px);
        box-shadow: 0 0 12px #2196F3, 0 0 24px #2196F3, 0 0 36px #4CAF50; /* نيون أزرق-أخضر */
    }

    /* لتحريك الأيقونة عند التحديد */
    .bottom-navbar .st-cr .st-cv .st-ce input[type="radio"]:checked + label .css-1dp5x4q {
        transform: translateY(-5px);
        transition: transform 0.3s ease;
    }

    /* --- خلفيات مختلفة للصفحات لتمييزها --- */
    [data-testid="stContainer"] > div:first-child { /* التحليل */
        background-color: rgba(227, 242, 253, 0.5); /* أزرق خفيف */
    }
    /* ملاحظة: للحاسبة والتتبع، يمكن تخصيص عبر if في الكود، لكن هذا عام */

    </style>
    """,
    unsafe_allow_html=True
)

# --- باقي الكود دون تغيير (الدوال والمنطق) ---
# (يمكنك نسخ باقي الكود من السكربت الأصلي هنا، حيث لم أغيره إلا في CSS)

# ... (استمر في نسخ الجزء الباقي من الكود الأصلي، مثل الدوال format_price، calculate_pnl_percentages، الـ session state، fetch_instruments، الشريط العلوي، إلخ.)

# في نهاية الكود، لتمييز الصفحات بشكل أفضل، يمكن إضافة هذا داخل كل if:
if selected_page == "📊 التحليل":
    st.markdown('<div style="background-color: rgba(227, 242, 253, 0.8); padding: 20px; border-radius: 15px;">', unsafe_allow_html=True)
    # ... محتوى التحليل ...
    st.markdown('</div>', unsafe_allow_html=True)

elif selected_page == "🧮 الحاسبة":
    st.markdown('<div style="background-color: rgba(232, 245, 233, 0.8); padding: 20px; border-radius: 15px;">', unsafe_allow_html=True)  # أخضر خفيف
    trading_calculator_app()
    st.markdown('</div>', unsafe_allow_html=True)

elif selected_page == "📈 المتتبع":
    st.markdown('<div style="background-color: rgba(255, 243, 224, 0.8); padding: 20px; border-radius: 15px;">', unsafe_allow_html=True)  # برتقالي خفيف
    live_market_tracker()
    st.markdown('</div>', unsafe_allow_html=True)
