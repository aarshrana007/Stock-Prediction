# ==============================
# Indian Stock Buy/Sell Predictor
# ==============================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import nltk
import re
from io import StringIO
from sentence_transformers import SentenceTransformer
from nltk.sentiment import SentimentIntensityAnalyzer

# ------------------------------
# Setup
# ------------------------------
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

st.set_page_config(
    page_title="Indian Stock Predictor",
    page_icon="📈",
    layout="centered"
)

# ------------------------------
# Styles
# ------------------------------
st.markdown("""
<style>
.big-font {font-size:48px; font-weight:bold; text-align:center;}
.pred-up {background:#d4edda; padding:25px; border-radius:15px; border:3px solid #28a745;}
.pred-down {background:#f8d7da; padding:25px; border-radius:15px; border:3px solid #dc3545;}
.news-item {padding:10px; margin-bottom:8px; background:#f0f2f6; border-radius:10px;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Indian Stock Buy / Sell Predictor")
st.caption("News + Sentiment + USD/INR + Historical Similarity")

# ------------------------------
# Load Embedding Model
# ------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ------------------------------
# Load Historical News
# ------------------------------
@st.cache_data
def load_historical_news():
    df = pd.read_csv("IndianFinancialNews.csv")
    df = df.dropna(subset=["Title"])
    df["Description"] = df["Title"].astype(str)
    df["date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    embeddings = embedder.encode(df["Description"].tolist(), batch_size=64)
    return df, np.array(embeddings).astype("float32")

df_hist, hist_embeddings = load_historical_news()

# ------------------------------
# Load Recent News (GitHub)
# ------------------------------
@st.cache_data(ttl=1800)
def load_recent_news():
    token = st.secrets.get("GITHUB_TOKEN", "")
    if not token:
        return pd.DataFrame()

    headers = {"Authorization": f"token {token}"}
    url = "https://api.github.com/repos/aarshrana007/Stock_Analysis/contents/news_data"

    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        return pd.DataFrame()

    dfs = []
    for f in res.json():
        if f["name"].endswith(".csv"):
            csv = requests.get(f["download_url"]).text
            df = pd.read_csv(StringIO(csv))

            # Normalize columns
            df["Description"] = (
                df.get("title", "").astype(str) + " " +
                df.get("summary", "").astype(str)
            ).str.strip()

            df["published_ist"] = df.get("published_ist", df.get("published", ""))
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    recent = pd.concat(dfs, ignore_index=True)

    # ------------------------------
    # DATA CLEANING (FIXES YOUR ERROR)
    # ------------------------------
    if "date" not in recent.columns:
        recent["date"] = ""

    def extract_date(val):
        if pd.isna(val) or str(val).strip() == "":
            return None
        parsed = pd.to_datetime(val, errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%d")
        match = re.search(r"\d{4}-\d{2}-\d{2}", str(val))
        return match.group(0) if match else None

    mask = recent["date"].isna() | (recent["date"].str.strip() == "")
    recent.loc[mask, "date"] = recent.loc[mask, "published_ist"].apply(extract_date)

    today = datetime.now().strftime("%Y-%m-%d")
    recent["date"] = recent["date"].fillna(today)

    return recent[["Description", "date"]]

recent_news_df = load_recent_news()

# ------------------------------
# Combine News + Embeddings
# ------------------------------
if not recent_news_df.empty:
    recent_emb = embedder.encode(recent_news_df["Description"].tolist(), batch_size=64)
    embeddings = np.vstack([hist_embeddings, recent_emb])
    df_combined = pd.concat(
        [df_hist[["Description", "date"]], recent_news_df],
        ignore_index=True
    )
else:
    embeddings = hist_embeddings
    df_combined = df_hist[["Description", "date"]]

# ------------------------------
# Market Snapshot
# ------------------------------
st.subheader("📊 Market Snapshot")

def safe_index(symbol):
    try:
        info = yf.Ticker(symbol).info
        return info.get("regularMarketPrice", "N/A"), info.get("regularMarketChangePercent", 0)
    except:
        return "N/A", 0

n_price, n_chg = safe_index("^NSEI")
usd_inr, usd_chg = safe_index("USDINR=X")

st.write(f"**NIFTY 50:** {n_price} ({n_chg:+.2f}%)")
st.write(f"**USD / INR:** {usd_inr} ({usd_chg:+.2f}%)")

# ------------------------------
# Predictor
# ------------------------------
st.subheader("🔮 Stock Prediction")

stock = st.text_input("Stock Symbol (e.g. AXISBANK)", value="AXISBANK").upper().strip()

if st.button("Predict 5-Day Move"):
    ticker = f"{stock}.NS"

    with st.spinner("Analyzing…"):

        # Sentiment
        texts = recent_news_df["Description"].tolist()[:50] if not recent_news_df.empty else df_combined["Description"].sample(20).tolist()
        sentiment = np.mean([sia.polarity_scores(t)["compound"] for t in texts])

        # USD/INR penalty
        usd_penalty = -0.8 if usd_chg > 0.3 else 0

        # Similarity search
        query_vec = embedder.encode([" ".join(texts)]).astype("float32")
        sim = np.dot(embeddings, query_vec.T).flatten()
        sim /= (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-8)

        top_idx = np.argsort(sim)[-8:][::-1]

        changes = []
        for i in top_idx:
            d = df_combined.iloc[i]["date"]
            try:
                data = yf.download(
                    ticker,
                    start=d,
                    end=(pd.to_datetime(d) + timedelta(days=10)).strftime("%Y-%m-%d"),
                    progress=False,
                    auto_adjust=True
                )
                if len(data) >= 3:
                    pct = (data["Close"].iloc[-1] / data["Close"].iloc[0] - 1) * 100
                    changes.append(pct)
            except:
                pass

        base_move = np.mean(changes) if changes else 0
        final_move = base_move + sentiment * 10 + usd_penalty

        direction = "UP 🟢" if final_move > 0 else "DOWN 🔴"
        css = "pred-up" if final_move > 0 else "pred-down"
        confidence = int(65 + min(abs(final_move), 5) * 6)

        st.markdown(
            f"""
            <div class="{css}">
                <div class="big-font">{direction} {abs(final_move):.2f}%</div>
                <p><b>Expected 5-Day Move for {stock}</b></p>
                <p>Confidence: {confidence}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.caption("✔ News cleaned ✔ USD/INR added ✔ Safe for Streamlit Cloud ✔ Kaggle ready")
