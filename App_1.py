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

# Custom CSS
st.markdown("""
<style>
    .big-font {font-size:50px !important; font-weight:bold; text-align:center;}
    .pred-up {background-color:#d4edda; padding:30px; border-radius:20px; border:4px solid #28a745; text-align:center;}
    .pred-down {background-color:#f8d7da; padding:30px; border-radius:20px; border:4px solid #dc3545; text-align:center;}
    .news-item {margin-bottom:12px; padding:12px; background-color:#f0f2f6; border-radius:10px;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Indian Stock Movement Predictor")
st.markdown("**AI-powered 5-day prediction using latest news sentiment + historical patterns**")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
embedder = load_model()

@st.cache_data
def load_data():
    df = pd.read_csv("IndianFinancialNews.csv")
    df = df.dropna(subset=['Title']).copy()
    df['Description'] = df['Title'].astype(str)
    df['date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    embeddings = embedder.encode(df['Description'].tolist(), batch_size=64)
    return df, np.array(embeddings).astype('float32')

df_hist, embeddings = load_data()

def get_recent_news(stock):
    url = f"https://www.google.com/search?q={stock}+stock+latest+news+site:moneycontrol.com+OR+site:economictimes.indiatimes.com+OR+site:livemint.com&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        headlines = [item.select_one('div[role="heading"]').get_text(strip=True) 
                    for item in soup.select('div.SoaBEf')[:8] if item.select_one('div[role="heading"]')]
        return headlines or ["No recent news found"]
    except:
        return ["News fetch failed"]

stock = st.text_input("Enter NSE Stock Symbol", value="RELIANCE", placeholder="e.g., SBI, TCS, HDFCBANK").upper().strip()

if st.button("🔮 Predict 5-Day Movement", type="primary"):
    ticker = f"{stock}.NS"
    
    with st.spinner("Analyzing news & historical patterns..."):
        news = get_recent_news(stock)
        
        st.subheader("📰 Recent News")
        for h in news[:5]:
            st.markdown(f"<div class='news-item'>{h}</div>", unsafe_allow_html=True)

        scores = [sia.polarity_scores(h)['compound'] for h in news]
        sentiment = np.mean(scores)
        mood = "Bullish 🟢" if sentiment > 0.05 else "Bearish 🔴" if sentiment < -0.05 else "Neutral ⚪"
        st.write(f"**Sentiment**: {mood} ({sentiment:+.3f})")

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

        final_move = np.mean(changes) + sentiment * 10 if changes else sentiment * 12
        confidence = int(68 + abs(sentiment)*25)
        direction = "UP 🟢" if final_move > 0 else "DOWN 🔴"
        pred_class = "pred-up" if final_move > 0 else "pred-down"

        st.markdown(f'<div class="{pred_class}"><div class="big-font">{direction} {abs(final_move):.1f}%</div>'
                    f'<h2>Expected 5-day move for {stock}</h2>'
                    f'<p><strong>Confidence: {confidence}%</strong></p></div>', unsafe_allow_html=True)

st.caption("Data: Indian Financial News (2003–2020) • Models: Sentence Transformers + VADER • Real-time News via Google")
