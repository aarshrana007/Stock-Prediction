import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Indian Stock Predictor", page_icon="📈", layout="centered")

# Custom CSS for a beautiful look
st.markdown("""
<style>
    .big-font {font-size:50px !important; font-weight:bold; text-align:center;}
    .pred-up {background-color:#d4edda; padding:30px; border-radius:20px; border:4px solid #28a745; text-align:center;}
    .pred-down {background-color:#f8d7da; padding:30px; border-radius:20px; border:4px solid #dc3545; text-align:center;}
    .news-item {margin-bottom:12px; padding:12px; background-color:#f0f2f6; border-radius:10px;}
    .market-box {padding:15px; border-radius:10px; text-align:center;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Indian Stock Movement Predictor")
st.markdown("**Real-time market data • Latest financial news • AI-powered 5-day predictions**")

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
embedder = load_model()

# Load historical news data
@st.cache_data
def load_historical():
    df = pd.read_csv("IndianFinancialNews.csv")  # Must be in repo
    df = df.dropna(subset=['Title']).copy()
    df['Description'] = df['Title'].astype(str)
    df['date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    embeddings = embedder.encode(df['Description'].tolist(), batch_size=64)
    return df, np.array(embeddings).astype('float32')

df_hist, embeddings = load_historical()

# === REAL-TIME MARKET SNAPSHOT ===
st.subheader("📊 Real-Time Market Snapshot")

@st.cache_data(ttl=300)  # Refresh every 5 minutes
def get_market_snapshot():
    try:
        nifty = yf.Ticker("^NSEI").info
        sensex = yf.Ticker("^BSESN").info
        n_price = nifty.get('regularMarketPrice', 'N/A')
        n_change = nifty.get('regularMarketChangePercent', 0)
        s_price = sensex.get('regularMarketPrice', 'N/A')
        s_change = sensex.get('regularMarketChangePercent', 0)
        return {"Nifty": (n_price, n_change), "Sensex": (s_price, s_change)}
    except:
        return {"Nifty": ("N/A", 0), "Sensex": ("N/A", 0)}

market = get_market_snapshot()

col1, col2 = st.columns(2)
with col1:
    n_price, n_change = market["Nifty"]
    color = "#d4edda" if n_change > 0 else "#f8d7da"
    st.markdown(f"<div class='market-box' style='background-color:{color}'>"
                f"<h3>Nifty 50</h3><h2>{n_price}</h2><p>{n_change:+.2f}%</p></div>", unsafe_allow_html=True)
with col2:
    s_price, s_change = market["Sensex"]
    color = "#d4edda" if s_change > 0 else "#f8d7da"
    st.markdown(f"<div class='market-box' style='background-color:{color}'>"
                f"<h3>Sensex</h3><h2>{s_price}</h2><p>{s_change:+.2f}%</p></div>", unsafe_allow_html=True)

# === TOP 10 LATEST FINANCIAL NEWS ===
st.subheader("📰 Top 10 Latest Indian Financial News")

@st.cache_data(ttl=3600)  # Refresh hourly
def get_top_financial_news():
    url = "https://www.google.com/search?q=india+financial+news+OR+stock+market+OR+nifty+OR+sensex+OR+rupee+OR+rbi+OR+sebi&tbm=nws&source=lnt&tbs=qdr:d"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        headlines = []
        for item in soup.select('div.SoaBEf')[:10]:
            title_elem = item.select_one('div[role="heading"]')
            if title_elem:
                headlines.append(title_elem.get_text(strip=True))
        return headlines if headlines else ["No latest financial news found"]
    except:
        return ["News temporarily unavailable"]

latest_news = get_top_financial_news()
for i, news in enumerate(latest_news, 1):
    st.markdown(f"<div class='news-item'><strong>{i}.</strong> {news}</div>", unsafe_allow_html=True)

st.markdown("---")

# === STOCK PREDICTOR ===
st.markdown("### 🔮 Predict Any NSE Stock")
stock = st.text_input("Enter Stock Symbol", value="RELIANCE", placeholder="e.g., SBI, TCS, HDFCBANK, INFY").upper().strip()

if st.button("Predict 5-Day Movement", type="primary"):
    ticker = f"{stock}.NS"

    with st.spinner("Fetching news & analyzing history..."):
        # Recent stock-specific news
        def get_recent_news(s):
            url = f"https://www.google.com/search?q={s}+stock+latest+news+site:moneycontrol.com+OR+site:economictimes.indiatimes.com+OR+site:livemint.com&tbm=nws"
            headers = {"User-Agent": "Mozilla/5.0"}
            try:
                r = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(r.text, 'html.parser')
                headlines = [item.select_one('div[role="heading"]').get_text(strip=True) 
                            for item in soup.select('div.SoaBEf')[:8] if item.select_one('div[role="heading"]')]
                return headlines or ["No recent news found"]
            except:
                return ["News unavailable"]

        news = get_recent_news(stock)
        st.subheader(f"📰 Recent News for {stock}")
        for h in news[:5]:
            st.write("• " + h)

        # Sentiment
        scores = [sia.polarity_scores(h)['compound'] for h in news]
        sentiment = np.mean(scores)
        mood = "Bullish 🟢" if sentiment > 0.05 else "Bearish 🔴" if sentiment < -0.05 else "Neutral ⚪"
        st.write(f"**Sentiment**: {mood} ({sentiment:+.3f})")

        # Similarity search
        query_vec = embedder.encode([" ".join(news)]).astype('float32')
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
        sim = np.dot(embeddings, query_vec.T).flatten() / (norms + 1e-8)
        top_idx = np.argsort(sim)[-10:][::-1]

        changes = []
        st.subheader("📊 Top Similar Historical Events")
        for idx in top_idx:
            row = df_hist.iloc[idx]
            try:
                data = yf.download(ticker, start=row['date'], 
                                 end=(pd.to_datetime(row['date']) + timedelta(days=15)).strftime('%Y-%m-%d'),
                                 progress=False, auto_adjust=True, timeout=10)
                if len(data) >= 4:
                    pct = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                    changes.append(pct)
                    st.write(f"**{row['date'][:10]}** → **{pct:+.1f}%** | {row['Description'][:100]}...")
            except:
                continue

        # Final prediction
        if changes:
            avg_change = np.mean(changes)
            final_move = avg_change + sentiment * 10
        else:
            final_move = sentiment * 12

        confidence = int(68 + abs(sentiment) * 25)
        direction = "UP 🟢" if final_move > 0 else "DOWN 🔴"
        pred_class = "pred-up" if final_move > 0 else "pred-down"

        st.markdown(f'<div class="{pred_class}"><div class="big-font">{direction} {abs(final_move):.1f}%</div>'
                    f'<h2>Expected 5-day move for {stock}</h2>'
                    f'<p><strong>Confidence: {confidence}%</strong></p></div>', unsafe_allow_html=True)

st.caption("Real-time data via Yahoo Finance • Historical News 2003–2020 • AI Models: Sentence Transformers + VADER Sentiment")
