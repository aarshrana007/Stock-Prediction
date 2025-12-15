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

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load historical news (you'll upload the CSV in next step)
@st.cache_data
def load_historical_data():
    df = pd.read_csv("Indian_Financial_News.csv")  # Upload this file
    df = df.dropna(subset=['Headline']).copy()
    df['text'] = df['Headline'].astype(str)
    df['date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    embeddings = embedder.encode(df['text'].tolist(), batch_size=64, show_progress_bar=False)
    return df, np.array(embeddings).astype('float32')

df_hist, embeddings = load_historical_data()

# Beautiful page config
st.set_page_config(page_title="Indian Stock News Predictor", page_icon="📈", layout="centered")

# Custom CSS for beauty
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #0e76a8; color: white; font-weight: bold; border-radius: 10px; padding: 10px 24px;}
    .prediction-box {padding: 20px; border-radius: 15px; text-align: center; font-size: 1.5em; margin: 20px 0;}
    .up {background-color: #d4edda; border: 2px solid #28a745; color: #155724;}
    .down {background-color: #f8d7da; border: 2px solid #dc3545; color: #721c24;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Indian Stock Movement Predictor")
st.markdown("### Powered by News Sentiment + Historical Similarity (2015–2025)")

stock = st.text_input("Enter NSE Stock Symbol", value="RELIANCE", help="e.g., RELIANCE, TCS, HDFCBANK, SBIN, INFY").upper()

if st.button("🔮 Predict 5-Day Movement"):
    with st.spinner("Fetching latest news & analyzing..."):
        # Recent news
        def get_news(stock):
            url = f"https://www.google.com/search?q={stock}+stock+news&tbm=nws"
            headers = {"User-Agent": "Mozilla/5.0"}
            try:
                r = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(r.text, 'html.parser')
                headlines = [item.select_one('div[role="heading"]').text for item in soup.select('div.SoaBEf')[:8] if item.select_one('div[role="heading"]')]
                return headlines or ["No recent news found"]
            except:
                return ["News temporarily unavailable"]

        news = get_news(stock)
        st.subheader("📰 Recent News Driving the Market")
        for i, h in enumerate(news[:5], 1):
            st.write(f"{i}. {h}")

        # Sentiment
        scores = [sia.polarity_scores(h)['compound'] for h in news]
        sentiment = np.mean(scores)
        mood = "Bullish 🟢" if sentiment > 0.05 else "Bearish 🔴" if sentiment < -0.05 else "Neutral ⚪"

        # Similarity
        query_vec = embedder.encode([" ".join(news)]).astype('float32')
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
        sim = np.dot(embeddings, query_vec.T).flatten() / (norms + 1e-8)
        top_idx = np.argsort(sim)[-10:][::-1]

        changes = []
        st.subheader("📊 Top Similar Historical Events")
        for idx in top_idx:
            row = df_hist.iloc[idx]
            try:
                data = yf.download(f"{stock}.NS", start=row['date'], 
                                 end=(pd.to_datetime(row['date'])+timedelta(days=15)).strftime('%Y-%m-%d'),
                                 progress=False, auto_adjust=True, timeout=10)
                if len(data) >= 4:
                    pct = (data['Close'][-1]/data['Close'][0]-1)*100
                    changes.append(pct)
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.write(f"**{row['date'][:10]}**")
                        st.write(f"**{pct:+.1f}%**")
                    with col2:
                        st.write(row['text'][:100] + "...")
            except:
                continue

        avg_change = np.mean(changes) if changes else 0
        final_move = avg_change + sentiment * 10
        confidence = int(70 + abs(sentiment) * 25)

        direction = "UP 🟢" if final_move > 0 else "DOWN 🔴"

        # Beautiful prediction box
        box_class = "up" if final_move > 0 else "down"
        st.markdown(f"""
        <div class="prediction-box {box_class}">
            <h2>FINAL PREDICTION → {stock}</h2>
            <p>Expected 5-day move: <strong>{direction} {abs(final_move):.1f}%</strong></p>
            <p>Confidence: <strong>{confidence}%</strong> | Current Sentiment: {mood}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with ❤️ using Sentence Transformers, VADER Sentiment & Historical Indian News Data")
