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
st.title("📈 Indian Stock Movement Predictor")
st.markdown("Enter any NSE stock symbol and get a 5-day prediction based on latest news + historical similarity")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
embedder = load_model()

# Upload the CSV when deploying (or commit to GitHub)
@st.cache_data
def load_data():
    df = pd.read_csv("IndianFinancialNews.csv")  # This file must be in the same folder
    df = df.dropna(subset=['Title']).copy()
    df['Description'] = df['Title'].astype(str)
    df['date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    embeddings = embedder.encode(df['Description'].tolist(), batch_size=64)
    return df, np.array(embeddings).astype('float32')

df_hist, embeddings = load_data()

stock = st.text_input("Enter Stock Symbol", value="RELIANCE", help="e.g., SBI, TCS, HDFCBANK, INFY").upper().strip()

if st.button("Predict 5-Day Movement", type="primary"):
    ticker = f"{stock}.NS"
    
    with st.spinner("Analyzing news & history..."):
        # Recent news
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
                return ["News fetch failed"]
        
        news = get_recent_news(stock)
        st.subheader("Recent News")
        for h in news[:5]:
            st.write("• " + h)

        # Sentiment
        scores = [sia.polarity_scores(h)['compound'] for h in news]
        sentiment = np.mean(scores)
        mood = "Bullish 🟢" if sentiment > 0.05 else "Bearish 🔴" if sentiment < -0.05 else "Neutral ⚪"
        st.write(f"**Sentiment**: {mood} ({sentiment:+.3f})")

        # Similarity
        query_vec = embedder.encode([" ".join(news)]).astype('float32')
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
        sim = np.dot(embeddings, query_vec.T).flatten() / (norms + 1e-8)
        top_idx = np.argsort(sim)[-10:][::-1]

        changes = []
        st.subheader("Top Similar Past Events")
        for idx in top_idx:
            row = df_hist.iloc[idx]
            try:
                data = yf.download(ticker, start=row['date'], 
                                 end=(pd.to_datetime(row['date'])+timedelta(days=15)).strftime('%Y-%m-%d'),
                                 progress=False, auto_adjust=True, timeout=10)
                if len(data) >= 4:
                    pct = (data['Close'].iloc[-1]/data['Close'].iloc[0]-1)*100
                    changes.append(pct)
                    st.write(f"**{row['date'][:10]}** → **{pct:+.1f}%** | {row['Description'][:100]}...")
            except:
                continue

        # Prediction
        if changes:
            avg_change = np.mean(changes)
            final_move = avg_change + sentiment * 10
        else:
            final_move = sentiment * 12

        confidence = int(68 + abs(sentiment)*25)
        direction = "UP 🟢" if final_move > 0 else "DOWN 🔴"

        color = "#d4edda" if final_move > 0 else "#f8d7da"
        border = "#28a745" if final_move > 0 else "#dc3545"
        st.markdown(f"""
        <div style="background-color:{color}; padding:30px; border-radius:20px; text-align:center; border:4px solid {border}">
            <h1>{direction} {abs(final_move):.1f}%</h1>
            <h2>Expected 5-day move for {stock}</h2>
            <p><strong>Confidence: {confidence}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

st.caption("Data: Indian Financial News (2003–2020) | Models: Sentence Transformers + VADER")
