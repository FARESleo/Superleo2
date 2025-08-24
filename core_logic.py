import numpy as np
import pandas as pd
from math import isnan
from data_fetchers import fetch_ohlcv, fetch_ticker, fetch_funding, fetch_oi, fetch_orderbook, fetch_trades, get_live_market_data
import streamlit as st
import requests
from datetime import datetime

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

def trading_calculator_app():
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

def live_market_tracker():
    st.markdown("---")
    st.header("📈 متتبع السوق اللحظي")
    st.write("استكشف العملات ذات التغيرات السعرية الكبيرة خلال الـ 24 ساعة الماضية.")

    cols = st.columns(3)
    with cols[0]:
        threshold = st.selectbox("عتبة التغيير (%):", options=[1, 5, 10, 20, 50, 100], index=4)
    with cols[1]:
        search_query = st.text_input("ابحث عن عملة...").lower()
    with cols[2]:
        filter_type = st.selectbox("الفلتر:", options=["الكل", "صعود فقط", "هبوط فقط"], format_func=lambda x: x)

    with st.spinner("جارٍ تحديث بيانات السوق اللحظية..."):
        all_coins_df = get_live_market_data()

    if all_coins_df.empty:
        st.warning("تعذر جلب البيانات. يرجى المحاولة لاحقاً.")
        return

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
    
    filtered_df['رمز العملة'] = filtered_df['symbol'].str.upper()
    
    filtered_df = filtered_df.sort_values(by='price_change_abs', ascending=False)
    
    if filtered_df.empty:
        st.info("لا توجد عملات مطابقة للمعايير المحددة.")
    else:
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
        
        display_df['السعر ($)'] = display_df['السعر ($)'].apply(lambda x: f"{x:,.4f}" if x > 0.001 else f"{x:,.8f}")
        display_df['التغيير (24س) %'] = display_df['التغيير (24س) %'].apply(lambda x: f"{x:,.2f}%")
        display_df['أعلى سعر (24س) ($)'] = display_df['أعلى سعر (24س) ($)'].apply(lambda x: f"{x:,.4f}" if x > 0.001 else f"{x:,.8f}")
        display_df['أدنى سعر (24س) ($)'] = display_df['أدنى سعر (24س) ($)'].apply(lambda x: f"{x:,.4f}" if x > 0.001 else f"{x:,.8f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption(f"آخر تحديث: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
