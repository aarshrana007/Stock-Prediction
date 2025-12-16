# =========================================================
# 📈 INDIAN STOCK BUY / SELL SIGNAL APP (PRODUCTION STYLE)
# =========================================================

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
import re

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Indian Stock Signal App", page_icon="📈", layout="centered")
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# ---------------------------------------------------------
# STYLING
# ---------------------------------------------------------
st.markdown("""
<style>
.big-font {font-size:42px; font-weight:bold; text-align:center;}
.signal-buy {background:#d4edda; padding:25px; border-radius:16px; border:3px solid #28a745;}
.signal-sell {background:#f8d7da; padding:25px; border-radius:16px; border:3px solid #dc3545;}
.signal-hold {background:#fff3cd; padding:25px; border-radius:16px; border:3px solid #ffc107;}
.news-item {margin-bottom:10px; padding:10px; background:#f0f2f6; border-radius:10px;}
.market-box {padding:15px; border-radius:12px; text-align:center;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Indian Stock Buy / Sell Signal")
st.caption("News + Historical Similarity + Price Trend (Explainable AI)")

# ---------------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ---------------------------------------------------------
# LOAD HISTORICAL NEWS DATA
# ---------------------------------------------------------
@st.cache_data
def load_historical_news():
    df = pd.read_csv("IndianFinancialNews.csv")
    df = df.dropna(subset=["Title"]).copy()
    df["Description"] = df["Title"].astype(str)
    df["date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    emb = embedder.encode(df["Description"].tolist(), batch_size=64)
    return df, np.array(emb).astype("float32")

df_hist, hist_embeddings = load_historical_news()

# ---------------------------------------------------------
# LOAD RECENT NEWS FROM PRIVATE GITHUB
# ---------------------------------------------------------
@st.cache_data(ttl=1800)
def load_recent_news():
    token = st.secrets.get("GITHUB_TOKEN")
    if not token:
        return pd.DataFrame()

    headers = {"Authorization": f"token {token}"}
    folder_url = "https://api.github.com/repos/aarshrana007/Stock_Analysis/contents/news_data"

    response = requests.get(folder_url, headers=headers)
    if response.status_code != 200:
        return pd.DataFrame()

    dfs = []
    for file in response.json():
        if file["name"].endswith(".csv"):
            csv_text = requests.get(file["download_url"], headers=headers).text
            df = pd.read_csv(StringIO(csv_text))
            if not df.empty:
                df["Description"] = (df.get("title", "") + " " + df.get("summary", "")).str.strip()
                df = df[df["Description"].str.len() > 20]
                dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    recent_df = pd.concat(dfs, ignore_index=True)
    return recent_df

recent_news_df = load_recent_news()

# ---------------------------------------------------------
# 🔧 DATA CLEANING: DATE FIX (YOUR LOGIC – PROFESSIONALIZED)
# ---------------------------------------------------------
if not recent_news_df.empty:
    recent_news_df["date"] = recent_news_df.get("date", "Unknown")

    def extract_date_from_published(published):
        if pd.isna(published) or str(published).strip() == "":
            return None
        parsed = pd.to_datetime(published, errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%d")
        match = re.search(r"\d{4}-\d{2}-\d{2}", str(published))
        return match.group(0) if match else None

    unknown_mask = (
        recent_news_df["date"].isna() |
        (recent_news_df["date"] == "Unknown") |
        (recent_news_df["date"].astype(str).str.strip() == "")
    )

    extracted = recent_news_df.loc[unknown_mask, "published_ist"].apply(extract_date_from_published)
    recent_news_df.loc[unknown_mask & extracted.notna(), "date"] = extracted

    today = datetime.now().strftime("%Y-%m-%d")
    recent_news_df.loc[
        recent_news_df["date"].isna() | (recent_news_df["date"] == "Unknown"),
        "date"
    ] = today

    recent_news_df["date"] = pd.to_datetime(recent_news_df["date"]).dt.strftime("%Y-%m-%d")

# ---------------------------------------------------------
# COMBINE DATASETS
# ---------------------------------------------------------
if not recent_news_df.empty:
    recent_emb = embedder.encode(recent_news_df["Description"].tolist(), batch_size=64)
    embeddings = np.vstack([hist_embeddings, recent_emb.astype("float32")])
    df_combined = pd.concat([df_hist, recent_news_df[["Description", "date"]]], ignore_index=True)
else:
    embeddings = hist_embeddings
    df_combined = df_hist

# ---------------------------------------------------------
# MARKET SNAPSHOT
# ---------------------------------------------------------
st.subheader("📊 Market Snapshot")
try:
    nifty = yf.Ticker("^NSEI").info
    sensex = yf.Ticker("^BSESN").info
    n_price, n_chg = nifty["regularMarketPrice"], nifty["regularMarketChangePercent"]
    s_price, s_chg = sensex["regularMarketPrice"], sensex["regularMarketChangePercent"]
except:
    n_price = n_chg = s_price = s_chg = "N/A"

c1, c2 = st.columns(2)
c1.markdown(f"<div class='market-box'>{n_price}<br>Nifty ({n_chg:+.2f}%)</div>", unsafe_allow_html=True)
c2.markdown(f"<div class='market-box'>{s_price}<br>Sensex ({s_chg:+.2f}%)</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# CORE ANALYTICS FUNCTIONS
# ---------------------------------------------------------
def compute_sentiment(texts):
    scores = [sia.polarity_scores(t)["compound"] for t in texts[:40]]
    avg = np.mean(scores) if scores else 0
    label = "Bullish 🟢" if avg > 0.15 else "Bearish 🔴" if avg < -0.15 else "Neutral ⚪"
    return avg, label

def technical_trend(symbol):
    df = yf.download(f"{symbol}.NS", period="6mo", progress=False)
    if len(df) < 50:
        return "Neutral ⚪"
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    return "Bullish 🟢" if df["MA20"].iloc[-1] > df["MA50"].iloc[-1] else "Bearish 🔴"

def historical_reaction(symbol, query_vec):
    sim = np.dot(embeddings, query_vec.T).flatten()
    top_idx = np.argsort(sim)[-8:]
    moves = []

    for idx in top_idx:
        row = df_combined.iloc[idx]
        try:
            data = yf.download(
                f"{symbol}.NS",
                start=row["date"],
                end=(pd.to_datetime(row["date"]) + timedelta(days=7)).strftime("%Y-%m-%d"),
                progress=False
            )
            if len(data) > 2:
                moves.append((data["Close"].iloc[-1] / data["Close"].iloc[0]) - 1)
        except:
            continue

    if not moves:
        return "Neutral ⚪"
    avg = np.mean(moves)
    return "Bullish 🟢" if avg > 0.01 else "Bearish 🔴"

def final_signal(sent, tech, hist):
    if sent > 0.2 and tech == hist == "Bullish 🟢":
        return "BUY 🟢", 78
    if sent < -0.2 and tech == hist == "Bearish 🔴":
        return "SELL 🔴", 78
    return "HOLD ⚪", 62

# ---------------------------------------------------------
# UI: PREDICTOR
# ---------------------------------------------------------
st.markdown("---")
stock = st.text_input("Enter NSE Stock Symbol", "SBI").upper()

if st.button("Generate Trade Signal"):
    with st.spinner("Analyzing..."):
        texts = recent_news_df["Description"].tolist() if not recent_news_df.empty else df_combined["Description"].sample(30).tolist()

        sent_score, sent_label = compute_sentiment(texts)
        tech = technical_trend(stock)
        query_vec = embedder.encode([" ".join(texts[:40])]).astype("float32")
        hist = historical_reaction(stock, query_vec)

        signal, confidence = final_signal(sent_score, tech, hist)

        css = "signal-buy" if "BUY" in signal else "signal-sell" if "SELL" in signal else "signal-hold"

        st.markdown(f"""
        <div class="{css}">
            <div class="big-font">{signal}</div>
            <p><strong>Confidence:</strong> {confidence}%</p>
            <p>📰 News Sentiment: {sent_label}</p>
            <p>📈 Price Trend: {tech}</p>
            <p>📊 Past Events: {hist}</p>
        </div>
        """, unsafe_allow_html=True)

st.caption("Educational use only — not financial advice")
