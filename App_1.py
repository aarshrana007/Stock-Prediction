# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from transformers import pipeline

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Stock Buy/Sell Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Stock Buy / Sell Decision Engine")
st.caption("FinBERT + Market Context + Heavy News Scraping")

# =========================
# LOAD FINBERT
# =========================
@st.cache_resource
def load_finbert():
    return pipeline(
        "sentiment-analysis",
        model="yiyanghkust/finbert-tone",
        tokenizer="yiyanghkust/finbert-tone"
    )

finbert = load_finbert()

# =========================
# MARKET SNAPSHOT
# =========================
st.subheader("🌍 Market Snapshot")

col1, col2, col3 = st.columns(3)

def safe_price(ticker):
    try:
        return yf.Ticker(ticker).info.get("regularMarketPrice", "N/A")
    except:
        return "N/A"

with col1:
    st.metric("Nifty 50", safe_price("^NSEI"))

with col2:
    st.metric("Sensex", safe_price("^BSESN"))

with col3:
    usd_inr = safe_price("USDINR=X")
    st.metric("USD / INR", usd_inr)

# =========================
# HEAVY NEWS SCRAPER (GOOGLE + RSS)
# =========================
def scrape_news(stock):
    articles = []

    query = f"{stock} stock news India"
    url = f"https://www.google.com/search?q={query}&tbm=nws"

    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=10)

    soup = BeautifulSoup(res.text, "html.parser")

    for item in soup.select("div.BNeawe.vvjwJb.AP7Wnd")[:8]:
        title = item.text
        articles.append({
            "title": title,
            "Description": title,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "source": "Google News"
        })

    return pd.DataFrame(articles)

# =========================
# DATA CLEANING (FIXES YOUR ERRORS)
# =========================
def clean_news_df(df):
    if df.empty:
        return df

    # Ensure required columns
    for col in ["date", "title", "Description"]:
        if col not in df.columns:
            df[col] = ""

    # Normalize date
    df["date"] = (
        pd.to_datetime(df["date"], errors="coerce")
        .fillna(pd.Timestamp.today())
        .dt.strftime("%Y-%m-%d")
    )

    df["Description"] = df["Description"].astype(str)
    df = df[df["Description"].str.len() > 20]

    return df.reset_index(drop=True)

# =========================
# USER INPUT
# =========================
stock = st.text_input("Enter Stock Symbol (NSE)", value="AXISBANK").upper()

# =========================
# LOAD / SCRAPE NEWS
# =========================
st.subheader("📰 News Analysis")

news_df = scrape_news(stock)
news_df = clean_news_df(news_df)

if news_df.empty:
    st.warning("No news found.")
    st.stop()

st.write(news_df[["date", "Description", "source"]].head(5))

# =========================
# FINBERT SENTIMENT
# =========================
texts = news_df["Description"].tolist()
results = finbert(texts)

sentiment_score = 0
explanations = []

for text, res in zip(texts, results):
    label = res["label"].lower()
    score = res["score"]

    if label == "positive":
        sentiment_score += score
    elif label == "negative":
        sentiment_score -= score

    explanations.append((label, score, text))

avg_sentiment = sentiment_score / len(results)

# =========================
# DECISION ENGINE
# =========================
if avg_sentiment > 0.15 and usd_inr != "N/A" and float(usd_inr) < 84:
    decision = "BUY 🟢"
elif avg_sentiment < -0.15:
    decision = "SELL 🔴"
else:
    decision = "HOLD ⚪"

# =========================
# OUTPUT
# =========================
st.subheader("📊 AI Decision")

st.metric("Final Decision", decision)
st.metric("Sentiment Score", f"{avg_sentiment:+.3f}")

# =========================
# MODEL THINKING (EXPLAINABILITY)
# =========================
st.subheader("🧠 Model Thinking (Why?)")

for lbl, sc, txt in explanations[:5]:
    emoji = "🟢" if lbl == "positive" else "🔴" if lbl == "negative" else "⚪"
    st.write(f"{emoji} **{lbl.upper()} ({sc:.2f})** — {txt}")

# =========================
# DISCLAIMER
# =========================
st.caption("⚠️ Educational only. Not financial advice.")
