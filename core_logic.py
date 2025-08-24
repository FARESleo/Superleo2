import numpy as np
import pandas as pd
from math import isnan
from datetime import datetime
from data_fetchers import fetch_ohlcv, fetch_ticker, fetch_funding, fetch_oi, fetch_orderbook, fetch_trades, get_live_market_data
import streamlit as st
import requests

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

def get_market_trend(ohlcv_df, ma_short=20, ma_long=50):
    if ohlcv_df.empty or len(ohlcv_df) < ma_long:
        return "Neutral"
    
    df = ohlcv_df.copy()
    df["ema_short"] = df["c"].ewm(span=ma_short, adjust=False).mean()
    df["ema_long"] = df["c"].ewm(span=ma_long, adjust=False).mean()
    
    if df["ema_short"].iloc[-1] > df["ema_long"].iloc[-1]:
        return "Bullish"
    elif df["ema_short"].iloc[-1] < df["ema_long"].iloc[-1]:
        return "Bearish"
    else:
        return "Neutral"

# New scoring system
def get_score_from_value(value, is_bullish, threshold, scaler):
    if value is None: return 0.5
    if is_bullish:
        score = 0.5 + np.tanh((value - threshold) * scaler) / 2
    else:
        score = 0.5 - np.tanh((value - threshold) * scaler) / 2
    return score

# New function to detect big candles with high volume
def detect_big_candle_volume(ohlcv_df, lookback_period=2, volume_multiplier=1.5):
    if ohlcv_df.empty or len(ohlcv_df) < lookback_period + 1:
        return None, None

    last_candles = ohlcv_df.iloc[-lookback_period:]
    avg_vol = ohlcv_df['vol'].iloc[:-lookback_period].mean()

    for i, candle in last_candles.iterrows():
        is_high_volume = candle['vol'] > avg_vol * volume_multiplier
        candle_size = abs(candle['c'] - candle['o'])
        avg_candle_size = ohlcv_df['c'].diff().abs().mean()
        is_big_candle = candle_size > avg_candle_size * 2

        if is_high_volume and is_big_candle:
            if candle['c'] > candle['o']:
                return "Bullish Big Candle", candle
            else:
                return "Bearish Big Candle", candle
    return None, None

# New function to calculate signal strength based on points
def calculate_signal_strength(metrics, weights):
    total_score = 0
    reason_list = []
    
    is_bullish = metrics["market_trend"] == "Bullish"

    # Core Metrics Scoring
    score_cvd = metrics["cvd"]
    if score_cvd is not None:
        if is_bullish and score_cvd > 0:
            total_score += 15
            reason_list.append("CVD إيجابي (قوة شراء).")
        elif not is_bullish and score_cvd < 0:
            total_score += 15
            reason_list.append("CVD سلبي (قوة بيع).")
        else:
            reason_list.append("CVD محايد أو معاكس للاتجاه.")

    score_ob = metrics["orderbook"]
    if score_ob is not None:
        if is_bullish and score_ob > 0:
            total_score += 15
            reason_list.append("دفتر الطلبات يميل للشراء.")
        elif not is_bullish and score_ob < 0:
            total_score += 15
            reason_list.append("دفتر الطلبات يميل للبيع.")
        else:
            reason_list.append("دفتر الطلبات محايد أو معاكس للاتجاه.")

    score_bt = metrics["backtest"]
    if score_bt is not None and score_bt > 0.6:
        total_score += 20
        reason_list.append(f"الاختبار الخلفي لديه معدل فوز مرتفع ({score_bt:.2f}).")
    elif score_bt is not None and score_bt < 0.4:
        total_score -= 10
        reason_list.append(f"الاختبار الخلفي لديه معدل فوز منخفض ({score_bt:.2f}).")

    # Smart Money Signal Scoring (High impact)
    if metrics["big_candle_signal"] == "Bullish Big Candle":
        total_score += 25
        reason_list.append("اكتشاف شمعة صعودية كبيرة بحجم تداول مرتفع.")
    elif metrics["big_candle_signal"] == "Bearish Big Candle":
        total_score -= 25
        reason_list.append("اكتشاف شمعة هبوطية كبيرة بحجم تداول مرتفع.")

    # Additional Factors
    if metrics["candle_signal"] == "Bullish Engulfing":
        total_score += 10
        reason_list.append("إشارة شمعة Bullish Engulfing.")
    elif metrics["candle_signal"] == "Bearish Engulfing":
        total_score -= 10
        reason_list.append("إشارة شمعة Bearish Engulfing.")
    
    # Check for Liquidity Traps (Advanced)
    # This is a simplified logic. In a real system, it's more complex.
    if metrics["market_trend"] == "Bullish" and metrics["oi"] is not None and metrics["oi"] > 1e9: # High OI as a proxy for liquidity
        if metrics["cvd"] < 0:
            # Bullish trend but CVD is bearish = a possible fakeout/bull trap
            total_score -= 10
            reason_list.append("تحذير: CVD سلبي على الرغم من الاتجاه الصاعد (قد يكون فخًا).")
    
    if metrics["market_trend"] == "Bearish" and metrics["oi"] is not None and metrics["oi"] > 1e9:
        if metrics["cvd"] > 0:
            # Bearish trend but CVD is bullish = a possible fakeout/bear trap
            total_score += 10
            reason_list.append("تحذير: CVD إيجابي على الرغم من الاتجاه الهابط (قد يكون فخًا).")

    # Final recommendation logic
    if total_score >= 70:
        recommendation = "LONG"
        strength = "Strong"
    elif total_score >= 50:
        recommendation = "LONG"
        strength = "Moderate"
    elif total_score <= -70: # A negative score for a strong bearish signal
        recommendation = "SHORT"
        strength = "Strong"
    elif total_score <= -50:
        recommendation = "SHORT"
        strength = "Moderate"
    else:
        recommendation = "Wait"
        strength = "Neutral"
        
    return total_score, recommendation, strength, reason_list

def compute_confidence(instId, bar="1H"):
    with st.spinner("Computing — gathering live data..."):
        ohlcv = fetch_ohlcv(instId, bar, limit=300)
        price = fetch_ticker(instId)
        funding = fetch_funding(instId)
        oi = fetch_oi(instId)
        bids, asks = fetch_orderbook(instId, depth=60)
        trades = fetch_trades(instId, limit=400)
    
    # Calculate metrics
    cvd = compute_cvd(trades)
    ob_imb = orderbook_imbalance(bids, asks)
    bt_win = simple_backtest_winrate(ohlcv, lookahead=6, stop_pct=0.01, rr=2.0)
    support, resistance = compute_support_resistance(ohlcv)
    candle_signal = detect_candle_signal(ohlcv, bar)
    market_trend = get_market_trend(ohlcv)
    big_candle_signal, big_candle_data = detect_big_candle_volume(ohlcv, lookback_period=5, volume_multiplier=2)

    # Prepare metrics dictionary for scoring
    metrics = {
        "cvd": cvd,
        "orderbook": ob_imb,
        "backtest": bt_win,
        "oi": oi,
        "market_trend": market_trend,
        "big_candle_signal": big_candle_signal,
        "candle_signal": candle_signal,
        "price": price,
    }
    
    # Compute the final confidence score and recommendation
    total_score, recommendation, strength, reason_list = calculate_signal_strength(metrics, None)
    
    confidence_pct = round(max(0, min(abs(total_score), 100)), 1) if not isnan(total_score) else None

    # Determine entry, target, and stop
    atr = calculate_atr(ohlcv, period=14)
    entry, target, stop = None, None, None
    
    if recommendation in ["LONG", "SHORT"] and atr is not None and price is not None and not isnan(atr):
        entry = price
        atr_factor = 2.0
        
        if recommendation == "LONG":
            target = round(entry + (atr * atr_factor), 6)
            stop = round(entry - (atr * atr_factor / 2), 6)
        else: # SHORT
            target = round(entry - (atr * atr_factor), 6)
            stop = round(entry + (atr * atr_factor / 2), 6)
    
    reason = "، ".join(reason_list) if reason_list else "لا توجد إشارات قوية بما يكفي لإعطاء توصية."
    
    # Map metrics to a 0-1 score for the UI
    ui_metrics_scores = {
        "cvd": get_score_from_value(cvd, recommendation == "LONG", threshold=0, scaler=1e-4) if cvd is not None else None,
        "orderbook": get_score_from_value(ob_imb, recommendation == "LONG", threshold=0, scaler=20) if ob_imb is not None else None,
        "funding": get_score_from_value(funding, recommendation == "LONG", threshold=0, scaler=500) if funding is not None else None,
        "oi": get_score_from_value(oi, recommendation == "LONG", threshold=1e8, scaler=1e-8) if oi is not None else None,
        "backtest": bt_win if bt_win is not None else None,
    }
    
    raw = {"price":price,"funding":funding,"oi":oi,"cvd":cvd,"orderbook_imbalance":ob_imb,"backtest_win":bt_win,"support":support,"resistance":resistance,"candle_signal":candle_signal, "atr":atr}

    return {"label": recommendation, "confidence_pct": confidence_pct, "recommendation": recommendation, "strength": strength, "entry": entry, "target": target, "stop": stop, "metrics": ui_metrics_scores, "weights": None, "raw": raw, "reason": reason}

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
