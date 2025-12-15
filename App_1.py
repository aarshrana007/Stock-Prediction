import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from sentence_transformers import SentenceTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from io import StringIO

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Indian Stock Predictor", page_icon="📈", layout="centered")

# Custom CSS for beauty
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
st.markdown("**Real-time market + Your live GitHub news + Historical patterns**")

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
embedder = load_model()

# Load historical data
@st.cache_data
def load_historical():
    df = pd.read_csv("IndianFinancialNews.csv")
    df = df.dropna(subset=['Title']).copy()
    df['Description'] = df['Title'].astype(str)
    df['date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    embeddings = embedder.encode(df['Description'].tolist(), batch_size=64)
    return df, np.array(embeddings).astype('float32')

df_hist, hist_embeddings = load_historical()

# Load recent news from your GitHub
@st.cache_data(ttl=1800)  # Refresh every 30 minutes
def load_recent_news_from_github():
    folder_url = "https://api.github.com/repos/aarshrana007/Stock_Analysis/contents/news_data"
    try:
        response = requests.get(folder_url)
        if response.status_code != 200:
            st.warning("Could not access GitHub recent news folder.")
            return pd.DataFrame()
        
        file_list = response.json()
        dfs = []
        for file in file_list:
            if file['name'].endswith('.csv'):
                csv_data = requests.get(file['download_url']).text
                df = pd.read_csv(StringIO(csv_data))
                if not df.empty:
                    df['Description'] = (df.get('title', '') + " " + df.get('summary', '')).str.strip()
                    df = df[df['Description'].str.len() > 20]
                    if not df.empty:
                        dfs.append(df)
        
        if dfs:
            recent_df = pd.concat(dfs, ignore_index=True)
            recent_df['date'] = pd.to_datetime(recent_df['published'], errors='coerce').dt.strftime('%Y-%m-%d')
            recent_df = recent_df.dropna(subset=['date'])
            return recent_df[['Description', 'title', 'link', 'date', 'source']]
    except Exception as e:
        st.error(f"Error loading recent news: {e}")
    return pd.DataFrame()

recent_news_df = load_recent_news_from_github()

# Combine embeddings
if not recent_news_df.empty:
    recent_emb = embedder.encode(recent_news_df['Description'].tolist(), batch_size=64)
    recent_emb = np.array(recent_emb).astype('float32')
    embeddings = np.vstack([hist_embeddings, recent_emb])
    df_combined = pd.concat([df_hist, recent_news_df[['Description', 'date']]], ignore_index=True)
    st.success(f"Loaded {len(recent_news_df)} recent news items from your GitHub!")
else:
    embeddings = hist_embeddings
    df_combined = df_hist
    st.info("Using historical data only (no recent GitHub news)")

# Real-Time Market Snapshot
st.subheader("📊 Real-Time Market Snapshot")
try:
    nifty = yf.Ticker("^NSEI").info
    sensex = yf.Ticker("^BSESN").info
    n_price = nifty.get('regularMarketPrice', 'N/A')
    n_change = nifty.get('regularMarketChangePercent', 0)
    s_price = sensex.get('regularMarketPrice', 'N/A')
    s_change = sensex.get('regularMarketChangePercent', 0)
except:
    n_price, n_change = "N/A", 0
    s_price, s_change = "N/A", 0

col1, col2 = st.columns(2)
with col1:
    color = "#d4edda" if n_change > 0 else "#f8d7da"
    st.markdown(f"<div class='market-box' style='background-color:{color}'>"
                f"<h3>Nifty 50</h3><h2>{n_price}</h2><p>{n_change:+.2f}%</p></div>", unsafe_allow_html=True)
with col2:
    color = "#d4edda" if s_change > 0 else "#f8d7da"
    st.markdown(f"<div class='market-box' style='background-color:{color}'>"
                f"<h3>Sensex</h3><h2>{s_price}</h2><p>{s_change:+.2f}%</p></div>", unsafe_allow_html=True)

# Top 10 Latest News from GitHub
st.subheader("📰 Top 10 Latest Financial News (Your GitHub Feed)")
if not recent_news_df.empty:
    latest = recent_news_df.sort_values(by='date', ascending=False).head(10)
    for _, row in latest.iterrows():
        st.markdown(f"<div class='news-item'><strong>{row['date']}</strong>: {row['Description']}</div>", unsafe_allow_html=True)
else:
    st.info("No recent news from GitHub")

st.markdown("---")

# Stock Predictor
st.markdown("### 🔮 Predict Any NSE Stock")
stock = st.text_input("Enter Stock Symbol", value="RELIANCE", placeholder="e.g., SBI, TCS").upper().strip()

if st.button("Predict 5-Day Movement", type="primary"):
    ticker = f"{stock}.NS"
    with st.spinner("Analyzing..."):
        # Use recent news for sentiment
        if not recent_news_df.empty:
            news_texts = recent_news_df['Description'].tolist()
        else:
            news_texts = df_combined['Description'].sample(20).tolist()
        
        scores = [sia.polarity_scores(t)['compound'] for t in news_texts]
        sentiment = np.mean(scores)
        mood = "Bullish 🟢" if sentiment > 0.05 else "Bearish 🔴" if sentiment < -0.05 else "Neutral ⚪"
        st.write(f"**Sentiment**: {mood} ({sentiment:+.3f})")

        query_vec = embedder.encode([" ".join(news_texts[:50])]).astype('float32')
        sim = np.dot(embeddings, query_vec.T).flatten() / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-8)
        top_idx = np.argsort(sim)[-10:][::-1]

        changes = []
        st.subheader("📊 Top Similar Historical Events")
        for idx in top_idx:
            row = df_combined.iloc[idx]
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

st.caption("Your live GitHub news feed + Historical data + AI prediction")
