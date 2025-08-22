-- coding: utf-8 --

""" Smart Money Scanner – V3

مصدر البيانات: OKX REST v5 (بدون API Key)

Tabs:

1. Chart: شموع + EMA/RSI/MACD


2. Derivatives: Funding + Open Interest


3. Liquidity: دفتر أوامر + خريطة بسيطة للتكدسات


4. CVD: Cumulative Volume Delta من /market/trades


5. Backtester: اختبار R:R (1:2 أو 1:3) على بيانات تاريخية مع قواعد دخول بسيطة


6. Risk: حاسبة مخاطرة (رافعة، وقف، هدف)




تشغيل: pip install streamlit requests pandas numpy plotly streamlit run smart_money_scanner_v3.py

ملاحظات:

OKX instId مثال: BTC-USDT-SWAP أو ETH-USDT-SWAP

لا نستخدم مفاتيح API. احترس من Rate Limit عند التحديث السريع. """


import requests import pandas as pd import numpy as np import streamlit as st import plotly.graph_objects as go from datetime import datetime, timezone

OKX_BASE = "https://www.okx.com/api/v5"

=========================

-------- Helpers --------

=========================

def okx_get(path: str, params: dict = None): url = f"{OKX_BASE}{path}" try: r = requests.get(url, params=params, timeout=10) r.raise_for_status() j = r.json() if j.get('code') != '0' and j.get('data') is None: return None return j except Exception: return None

@st.cache_data(show_spinner=False) def get_ohlcv(instId: str = "BTC-USDT-SWAP", bar: str = "1H", limit: int = 400) -> pd.DataFrame: j = okx_get("/market/candles", {"instId": instId, "bar": bar, "limit": limit}) if not j or 'data' not in j: return pd.DataFrame() df = pd.DataFrame(j['data'], columns=[ "ts","o","h","l","c","vol","volCcy","volCcyQuote","confirm" ]) for col in ["o","h","l","c","vol"]: df[col] = df[col].astype(float) df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True) df = df.iloc[::-1].reset_index(drop=True) return df

@st.cache_data(show_spinner=False) def get_funding(instId: str = "BTC-USDT-SWAP"): j = okx_get("/public/funding-rate", {"instId": instId}) try: return float(j['data'][0]['fundingRate']) except Exception: return None

@st.cache_data(show_spinner=False) def get_open_interest(instId: str = "BTC-USDT-SWAP"): j = okx_get("/public/open-interest", {"instId": instId}) try: return float(j['data'][0]['oi']) except Exception: return None

@st.cache_data(show_spinner=False) def get_orderbook(instId: str = "BTC-USDT-SWAP", depth: int = 40): j = okx_get("/market/books", {"instId": instId, "sz": depth}) if not j or 'data' not in j: return pd.DataFrame(), pd.DataFrame() ob = j['data'][0] bids = pd.DataFrame(ob.get('bids', []), columns=["price","size","liq"]).astype(float) asks = pd.DataFrame(ob.get('asks', []), columns=["price","size","liq"]).astype(float) return bids, asks

@st.cache_data(show_spinner=False) def get_trades(instId: str = "BTC-USDT-SWAP", limit: int = 200): j = okx_get("/market/trades", {"instId": instId, "limit": limit}) if not j or 'data' not in j: return pd.DataFrame() df = pd.DataFrame(j['data']) # columns: instId, tradeId, px, sz, side, ts df['px'] = df['px'].astype(float) df['sz'] = df['sz'].astype(float) df['ts'] = pd.to_datetime(df['ts'].astype(int), unit='ms', utc=True) df = df.iloc[::-1].reset_index(drop=True) return df

=========================

----- Indicators --------

=========================

def ema(series: pd.Series, period=20): return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period=14): delta = series.diff() gain = delta.clip(lower=0).rolling(period).mean() loss = (-delta.clip(upper=0)).rolling(period).mean() rs = gain / (loss + 1e-9) return 100 - (100/(1+rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9): efast = ema(series, fast) eslow = ema(series, slow) line = efast - eslow sig = ema(line, signal) hist = line - sig return line, sig, hist

=========================

---- Pattern Finder ------

=========================

def detect_engulfing(df: pd.DataFrame): # بسيط: Bullish إذا جسم الشمعة يبتلع السابقة والعكس للـ Bearish eng_bull = [] eng_bear = [] for i in range(1, len(df)): o1,c1 = df['o'].iloc[i-1], df['c'].iloc[i-1] o2,c2 = df['o'].iloc[i], df['c'].iloc[i] if c2>o2 and o2<=min(o1,c1) and c2>=max(o1,c1): eng_bull.append(df['ts'].iloc[i]) if c2<o2 and o2>=max(o1,c1) and c2<=min(o1,c1): eng_bear.append(df['ts'].iloc[i]) return eng_bull, eng_bear

=========================

------- Backtester -------

=========================

def backtest_rr(df: pd.DataFrame, rr: int = 2, stop_pct: float = 0.01, lookahead: int = 6): """اختبار بسيط: دخول لونغ عندما EMA20>EMA50 و RSI بين 45-70. يختبر خلال lookahead شموع من بعد الدخول؛ لو high وصل الهدف قبل low وصل الوقف → ربح، والعكس خسارة. محافظ على التحفظ: إذا نفس الشمعة حققت الاثنين نعدّها خسارة (أسوأ حالة). """ if len(df) < 60: return {"trades":0,"wins":0,"losses":0,"winrate":0.0} df = df.copy() df['EMA20'] = ema(df['c'], 20) df['EMA50'] = ema(df['c'], 50) df['RSI'] = rsi(df['c']) wins=losses=0 for i in range(50, len(df)-lookahead-1): row = df.iloc[i] cond_long = row['EMA20']>row['EMA50'] and 45<=row['RSI']<=70 if not cond_long: continue entry = row['c'] stop = entry*(1-stop_pct) target = entry*(1+stop_pctrr) future = df.iloc[i+1:i+1+lookahead] hit_target=False; hit_stop=False for _, r in future.iterrows(): if r['low']<=stop: hit_stop=True if r['high']>=target: hit_target=True if hit_stop and hit_target: break if hit_target and not hit_stop: wins+=1 else: losses+=1 total=wins+losses return { "trades": total, "wins": wins, "losses": losses, "winrate": (wins/total100.0) if total>0 else 0.0 }

=========================

--------- CVD -----------

=========================

def compute_cvd(trades_df: pd.DataFrame): if trades_df.empty: return trades_df, 0.0 # side: 'buy' أو 'sell' حسب OKX trades_df = trades_df.copy() trades_df['signed_vol'] = np.where(trades_df['side'].str.lower()=="buy", trades_df['sz'], -trades_df['sz']) trades_df['cvd'] = trades_df['signed_vol'].cumsum() return trades_df, trades_df['cvd'].iloc[-1]

=========================

--------- UI -------------

=========================

st.set_page_config(page_title="Smart Money Scanner V3", layout="wide") st.title("🧠 Smart Money Scanner V3 – OKX")

with st.sidebar: st.header("⚙️ الإعدادات") instId = st.text_input("Instrument (OKX)", value="BTC-USDT-SWAP") bar = st.selectbox("Timeframe", ["1m","5m","15m","1H","4H","1D"], index=2) depth = st.slider("Depth (Orderbook levels)", 10, 100, 40, 10) st.caption("أمثلة: BTC-USDT-SWAP, ETH-USDT-SWAP, SOL-USDT-SWAP")

Tabs

chart_tab, deriv_tab, liq_tab, cvd_tab, bt_tab, risk_tab = st.tabs([ "Chart","Derivatives","Liquidity","CVD","Backtester","Risk" ])

------ Chart ------

with chart_tab: df = get_ohlcv(instId, bar, 400) if df.empty: st.error("تعذر جلب الشموع") else: df['EMA20'] = ema(df['c'], 20) df['EMA50'] = ema(df['c'], 50) df['RSI'] = rsi(df['c']) m_line, m_sig, m_hist = macd(df['c']) df['MACD']=m_line; df['MACDsig']=m_sig; df['Mhist']=m_hist

fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['ts'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name='Price'))
    fig.add_trace(go.Scatter(x=df['ts'], y=df['EMA20'], name='EMA20'))
    fig.add_trace(go.Scatter(x=df['ts'], y=df['EMA50'], name='EMA50'))
    fig.update_layout(height=420, margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("RSI / MACD")
    rcol1, rcol2 = st.columns(2)
    with rcol1:
        st.line_chart(df.set_index('ts')['RSI'])
    with rcol2:
        macdf = pd.DataFrame({
            'ts': df['ts'],
            'MACD': df['MACD'],
            'Signal': df['MACDsig']
        }).set_index('ts')
        st.line_chart(macdf)

    # أنماط بسيطة
    bulls, bears = detect_engulfing(df)
    st.caption(f"Bullish engulfing: {len(bulls)} | Bearish engulfing: {len(bears)} (آخر {len(df)} شمعة)")

------ Derivatives ------

with deriv_tab: col1, col2 = st.columns(2) funding = get_funding(instId) oi = get_open_interest(instId) col1.metric("Funding Rate", f"{funding:.4%}" if funding is not None else "N/A") col2.metric("Open Interest", f"{oi:,.0f}" if oi is not None else "N/A") st.caption("قراءة سريعة: Funding>0 يعني ضغط لونغ يدفع تمويلاً للشورت؛ Funding<0 العكس.")

------ Liquidity ------

with liq_tab: bids, asks = get_orderbook(instId, depth) if bids.empty or asks.empty: st.error("لا توجد بيانات دفتر أوامر") else: # ترتيب حسب السعر (بشكل تصاعدي) bids = bids.sort_values('price') asks = asks.sort_values('price') liq_fig = go.Figure() liq_fig.add_trace(go.Bar(x=bids['price'], y=bids['size'], name='Bids')) liq_fig.add_trace(go.Bar(x=asks['price'], y=asks['size'], name='Asks')) liq_fig.update_layout(barmode='overlay', height=420, margin=dict(l=10,r=10,t=20,b=10)) st.plotly_chart(liq_fig, use_container_width=True) st.caption("مناطق الأحجام الكبيرة غالبًا نقاط سيولة/فخاخ محتملة.")

------ CVD ------

with cvd_tab: tdf = get_trades(instId, 400) if tdf.empty: st.error("تعذر جلب الصفقات الأخيرة") else: tdf, last_cvd = compute_cvd(tdf) st.line_chart(tdf.set_index('ts')['cvd']) st.metric("آخر قيمة CVD", f"{last_cvd:,.2f}") st.caption("CVD يرتفع = طلب شرائي مهيمن؛ ينخفض = عرض بيعي مهيمن.")

------ Backtester ------

with bt_tab: if df.empty: st.info("افتح تبويب Chart أولًا") else: rr = st.selectbox("Risk:Reward", [2,3], index=0) stop_pct = st.slider("Stop % (حركة سعر)", 0.5, 5.0, 1.0, 0.1) / 100 lookahead = st.slider("Lookahead شموع", 3, 15, 6, 1) res = backtest_rr(df[['ts','o','h','l','c']].copy(), rr=rr, stop_pct=stop_pct, lookahead=lookahead) st.write(res) if res['trades']>0: st.progress(min(1.0, res['winrate']/100.0)) st.caption("قواعد الدخول بسيطة ويمكن تعديلها. الباكتيست محافظ (يرجّح الخسارة عند التعارض داخل نفس الشمعة).")

------ Risk ------

with risk_tab: st.subheader("حاسبة إدارة المخاطر") equity = st.number_input("رصيدك (USDT)", min_value=1.0, value=21.0, step=1.0) risk_pct = st.slider("نسبة المخاطرة/صفقة %", 0.5, 5.0, 2.0, 0.5) stop_move = st.slider("المسافة إلى الوقف %", 0.3, 5.0, 1.5, 0.1) leverage = st.slider("الرافعة", 1, 25, 10, 1)

risk_amount = equity * (risk_pct/100.0)
position_notional = risk_amount / (stop_move/100.0)
position_with_leverage = position_notional * leverage

tp_1_2 = stop_move * 2
tp_1_3 = stop_move * 3

st.write({
    'مبلغ المخاطرة/صفقة (USDT)': round(risk_amount, 3),
    'حجم المركز النظري (بدون رافعة)': round(position_notional, 3),
    'حجم المركز مع الرافعة': round(position_with_leverage, 3),
    'هدف 1:2 (%)': round(tp_1_2, 2),
    'هدف 1:3 (%)': round(tp_1_3, 2)
})
st.caption("اضبط الدخول بحيث لو ضرب الوقف تخسر فقط المبلغ المحدد. بعد ربح +1R يمكنك نقل الوقف للتعادل.")

