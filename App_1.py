import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import re
from io import StringIO
from datetime import datetime, timedelta, timezone
import xml.etree.ElementTree as ET

# ----------------------------
# Optional: FinBERT sentiment
# ----------------------------
FINBERT_AVAILABLE = True
try:
    from transformers import pipeline
except Exception:
    FINBERT_AVAILABLE = False

# VADER fallback
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()


# ============================
# Streamlit page config
# ============================
st.set_page_config(page_title="AI Stock Buy/Sell Decision Engine", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .big-title {font-size:42px; font-weight:800; margin-bottom: 0px;}
    .subtle {color:#6b7280; margin-top: 0px;}
    .card {padding:18px; border-radius:14px; border:1px solid #e5e7eb; background:white;}
    .kpi {font-size:28px; font-weight:800;}
    .kpi-label {color:#6b7280; font-size:14px;}
    .pill {display:inline-block; padding:6px 10px; border-radius:999px; font-size:12px; border:1px solid #e5e7eb; margin-right:6px;}
    .good {background:#d1fae5; border-color:#10b981;}
    .bad {background:#fee2e2; border-color:#ef4444;}
    .neutral {background:#f3f4f6; border-color:#9ca3af;}
    .reason {margin: 6px 0px;}
    .small {font-size:12px; color:#6b7280;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">AI Stock Buy / Sell Decision Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">FinBERT + Market Context + News (CSV + Google News RSS fallback)</div>', unsafe_allow_html=True)
st.write("")


# ============================
# Helpers
# ============================
def safe_str(x) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)

def normalize_symbol(sym: str) -> str:
    return re.sub(r"\s+", "", sym.upper().strip())

def get_today_date_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def parse_any_date(val):
    """Robust datetime parsing. Returns pd.Timestamp or NaT."""
    if val is None:
        return pd.NaT
    s = str(val).strip()
    if s == "" or s.lower() in ["unknown", "na", "n/a", "none"]:
        return pd.NaT
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    if pd.isna(ts):
        # Try extract YYYY-MM-DD inside string
        m = re.search(r"\d{4}-\d{2}-\d{2}", s)
        if m:
            ts = pd.to_datetime(m.group(0), errors="coerce", utc=True)
    return ts

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure standard columns exist: date, title, link, source, Description"""
    df = df.copy()
    # Try to find best title column
    title_candidates = ["title", "Title", "headline", "Headline"]
    summary_candidates = ["summary", "Summary", "description", "Description", "body", "Body", "content"]
    link_candidates = ["link", "url", "URL"]
    source_candidates = ["source", "Source", "publisher", "Publisher"]
    date_candidates = ["date", "Date", "published", "published_ist", "publishedAt", "time", "datetime"]

    def pick_col(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    tcol = pick_col(title_candidates)
    scol = pick_col(summary_candidates)
    lcol = pick_col(link_candidates)
    srccol = pick_col(source_candidates)
    dcol = pick_col(date_candidates)

    if "title" not in df.columns:
        df["title"] = df[tcol].astype(str) if tcol else ""
    if "link" not in df.columns:
        df["link"] = df[lcol].astype(str) if lcol else ""
    if "source" not in df.columns:
        df["source"] = df[srccol].astype(str) if srccol else ""

    # Build Description from title + summary if possible
    if "Description" not in df.columns:
        title_part = df[tcol].astype(str) if tcol else df["title"].astype(str)
        summary_part = df[scol].astype(str) if (scol and scol != tcol) else ""
        df["Description"] = (title_part + " " + summary_part).str.strip()
    else:
        df["Description"] = df["Description"].astype(str)

    # Create a raw date column
    if "raw_date" not in df.columns:
        df["raw_date"] = df[dcol].astype(str) if dcol else ""

    return df

def clean_news_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans any news dataframe structure into:
    columns: date (YYYY-MM-DD), title, link, source, Description
    Fixes unknown/missing dates using published_ist/published/raw_date else today.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "title", "link", "source", "Description"])

    df = ensure_columns(df)

    # Parse dates from multiple potential columns if present
    # Priority: date -> published_ist -> published -> raw_date
    candidates = []
    for col in ["date", "published_ist", "published", "Date", "raw_date"]:
        if col in df.columns:
            candidates.append(col)

    parsed = None
    for col in candidates:
        series = df[col].apply(parse_any_date)
        if parsed is None:
            parsed = series
        else:
            # fill missing with next candidate
            parsed = parsed.fillna(series)

    df["date_ts"] = parsed
    # Final fallback: today
    df.loc[df["date_ts"].isna(), "date_ts"] = pd.Timestamp(get_today_date_str(), tz="UTC")

    # Standardize date string
    df["date"] = df["date_ts"].dt.strftime("%Y-%m-%d")

    # Clean text
    df["title"] = df["title"].apply(safe_str).str.strip()
    df["Description"] = df["Description"].apply(safe_str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["link"] = df["link"].apply(safe_str).str.strip()
    df["source"] = df["source"].apply(safe_str).str.strip()

    # Drop very short rows
    df = df[df["Description"].str.len() > 15].copy()

    # Deduplicate by (date, Description)
    df = df.drop_duplicates(subset=["date", "Description"]).reset_index(drop=True)
    return df[["date", "title", "link", "source", "Description"]]


# ============================
# Google News RSS fallback
# ============================
def google_news_rss_fetch(query: str, max_items: int = 20) -> pd.DataFrame:
    """
    Safe, lightweight RSS fetch (not heavy scraping).
    Returns columns: date,title,link,source,Description
    """
    q = requests.utils.quote(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
    except Exception:
        return pd.DataFrame(columns=["date", "title", "link", "source", "Description"])

    try:
        root = ET.fromstring(r.text)
    except Exception:
        return pd.DataFrame(columns=["date", "title", "link", "source", "Description"])

    items = []
    for item in root.findall(".//item")[:max_items]:
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        pubDate = item.findtext("pubDate") or ""
        source = ""
        # Some RSS items include <source>
        src_el = item.find("source")
        if src_el is not None and src_el.text:
            source = src_el.text

        ts = parse_any_date(pubDate)
        if pd.isna(ts):
            ts = pd.Timestamp(get_today_date_str(), tz="UTC")
        date_str = ts.strftime("%Y-%m-%d")

        items.append({
            "date": date_str,
            "title": title,
            "link": link,
            "source": source,
            "Description": title
        })

    df = pd.DataFrame(items)
    return clean_news_df(df)


# ============================
# Market snapshot
# ============================
@st.cache_data(ttl=300)
def market_snapshot():
    def get_info(ticker):
        try:
            info = yf.Ticker(ticker).info
            price = info.get("regularMarketPrice", None)
            chg = info.get("regularMarketChangePercent", None)
            return price, chg
        except Exception:
            return None, None

    nifty_p, nifty_c = get_info("^NSEI")
    sensex_p, sensex_c = get_info("^BSESN")

    # USD/INR on Yahoo is often "INR=X"
    usd_p, usd_c = get_info("INR=X")
    return {
        "nifty": (nifty_p, nifty_c),
        "sensex": (sensex_p, sensex_c),
        "usd_inr": (usd_p, usd_c),
    }


# ============================
# FinBERT sentiment
# ============================
@st.cache_resource
def load_finbert():
    if not FINBERT_AVAILABLE:
        return None
    try:
        # ProsusAI/finbert is commonly used and returns POSITIVE/NEGATIVE/NEUTRAL
        nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        return nlp
    except Exception:
        return None

finbert = load_finbert()

def score_sentiment(texts):
    """
    Returns:
      overall_score (float): positive minus negative average
      label (str): Bullish / Bearish / Neutral
      per_item (list): [{text,label,conf,score}]
    """
    per_item = []
    if not texts:
        return 0.0, "Neutral ⚪", per_item

    texts = [t[:512] for t in texts]  # keep it safe

    if finbert is not None:
        res = finbert(texts)
        for t, r in zip(texts, res):
            lab = (r.get("label") or "").upper()
            conf = float(r.get("score") or 0.0)
            if "POS" in lab:
                s = +conf
                pretty = "Positive 🟢"
            elif "NEG" in lab:
                s = -conf
                pretty = "Negative 🔴"
            else:
                s = 0.0
                pretty = "Neutral ⚪"
            per_item.append({"text": t, "label": pretty, "conf": conf, "score": s})
    else:
        # VADER fallback
        for t in texts:
            conf = float(sia.polarity_scores(t)["compound"])
            # map to pseudo confidence
            s = conf
            if conf > 0.05:
                pretty = "Positive 🟢"
            elif conf < -0.05:
                pretty = "Negative 🔴"
            else:
                pretty = "Neutral ⚪"
            per_item.append({"text": t, "label": pretty, "conf": abs(conf), "score": s})

    overall = float(np.mean([x["score"] for x in per_item])) if per_item else 0.0
    if overall > 0.10:
        lbl = "Bullish 🟢"
    elif overall < -0.10:
        lbl = "Bearish 🔴"
    else:
        lbl = "Neutral ⚪"
    return overall, lbl, per_item


# ============================
# “Decision Engine”
# ============================
def decision_engine(sent_score, usd_inr_change_pct):
    """
    Simple rule-based overlay:
    - sentiment is main driver
    - USD/INR rising (rupee weaker) adds slight negative bias for banks (risk-off)
    """
    reasons = []
    # Weighting
    w_sent = 1.0
    w_fx = 0.30  # small modifier

    fx_signal = 0.0
    if usd_inr_change_pct is not None:
        # If USD/INR is up -> INR weaker -> mild negative for risk sentiment
        if usd_inr_change_pct > 0.15:
            fx_signal = -1.0
            reasons.append(f"USD/INR is rising (+{usd_inr_change_pct:.2f}%) → INR weaker → mild risk-off bias for banks.")
        elif usd_inr_change_pct < -0.15:
            fx_signal = +0.5
            reasons.append(f"USD/INR is falling ({usd_inr_change_pct:.2f}%) → INR stronger → mild positive sentiment.")
        else:
            reasons.append("USD/INR change is small → neutral FX impact.")
    else:
        reasons.append("USD/INR data not available → FX impact skipped.")

    score = (w_sent * sent_score) + (w_fx * fx_signal)

    # Decision thresholds
    if score >= 0.15:
        action = "BUY 🟢"
    elif score <= -0.15:
        action = "SELL 🔴"
    else:
        action = "HOLD ⚪"

    reasons.insert(0, f"Sentiment score = {sent_score:+.3f} (main driver).")
    reasons.append(f"Final decision score = {score:+.3f} → {action}")
    return action, score, reasons


# ============================
# UI: Market Snapshot
# ============================
snap = market_snapshot()
nifty_p, nifty_c = snap["nifty"]
sensex_p, sensex_c = snap["sensex"]
usd_p, usd_c = snap["usd_inr"]

st.subheader("🌍 Market Snapshot")

c1, c2, c3 = st.columns(3)

def kpi_card(col, title, price, chg):
    with col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-label">{title}</div>', unsafe_allow_html=True)
        if price is None:
            st.markdown('<div class="kpi">N/A</div>', unsafe_allow_html=True)
            st.markdown('<div class="small">Yahoo may rate-limit sometimes.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="kpi">{price}</div>', unsafe_allow_html=True)
            if chg is None:
                st.markdown('<span class="pill neutral">Change N/A</span>', unsafe_allow_html=True)
            else:
                cls = "good" if chg > 0 else "bad" if chg < 0 else "neutral"
                st.markdown(f'<span class="pill {cls}">{chg:+.2f}%</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

kpi_card(c1, "Nifty 50", nifty_p, nifty_c)
kpi_card(c2, "Sensex", sensex_p, sensex_c)
kpi_card(c3, "USD / INR", usd_p, usd_c)

st.write("")


# ============================
# Input: News CSV
# ============================
st.subheader("📰 News Input")

uploaded = st.file_uploader("Upload your news CSV (optional). If not provided, app will rely on Google News RSS fallback.", type=["csv"])

raw_df = pd.DataFrame()
if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
        st.success(f"Loaded file with {len(raw_df):,} rows and columns: {list(raw_df.columns)}")
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
else:
    st.info("No CSV uploaded. The app will use Google News RSS when needed.")

recent_news_df = clean_news_df(raw_df) if not raw_df.empty else pd.DataFrame(columns=["date","title","link","source","Description"])


# ============================
# Stock Analyzer
# ============================
st.subheader("🔍 Stock Analyzer")

symbol = normalize_symbol(st.text_input("Enter NSE Symbol (example: AXISBANK, SBIN, RELIANCE)", value="AXISBANK"))

# Simple name map (extend as needed)
name_map = {
    "AXISBANK": ["axis bank", "axisbank", "axis"],
    "SBIN": ["state bank of india", "sbi", "state bank"],
    "SBI": ["state bank of india", "sbi", "state bank"],
    "RELIANCE": ["reliance", "ril"],
    "HDFCBANK": ["hdfc bank"],
    "ICICIBANK": ["icici bank"],
    "TCS": ["tcs", "tata consultancy"],
    "INFY": ["infosys"]
}

keywords = [symbol]
keywords += name_map.get(symbol, [])
pattern = "|".join([re.escape(k) for k in keywords if k])

def get_related_news(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    mask = df["Description"].str.contains(pattern, case=False, na=False)
    out = df[mask].copy()
    out = out.sort_values("date", ascending=False)
    return out

related = get_related_news(recent_news_df)

# If not found in CSV → fetch from Google News RSS
used_source = "CSV"
if related.empty:
    used_source = "Google News RSS"
    related = google_news_rss_fetch(f"{symbol} stock India", max_items=25)
    related = get_related_news(related)

st.markdown(f"**News source used:** `{used_source}`")

if related.empty:
    st.warning("No news found even after RSS. Try a different symbol or keyword.")
    st.stop()

# Show latest news used
st.write("")
st.markdown("### Latest news used for decision")
show_n = st.slider("How many headlines to use?", min_value=5, max_value=25, value=12, step=1)
headlines = related.head(show_n)["Description"].tolist()

for i, row in enumerate(related.head(10).itertuples(), start=1):
    st.markdown(f"- **{row.date}** — {row.Description[:180]}{'...' if len(row.Description) > 180 else ''}")

# Run sentiment
sent_score, sent_label, per_item = score_sentiment(headlines)

# USD/INR condition (use daily change percent from snapshot)
usd_inr_change = usd_c  # percent
action, final_score, reasons = decision_engine(sent_score, usd_inr_change)

st.write("")
st.markdown("### Model Thinking (What influenced the decision)")

left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"**Overall sentiment:** `{sent_label}`  \n**Sentiment score:** `{sent_score:+.3f}`", unsafe_allow_html=True)
    st.markdown(f"**Decision:** **{action}**  \n**Decision score:** `{final_score:+.3f}`", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Reasons:**")
    for r in reasons:
        st.markdown(f"- {r}")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Headline-by-headline sentiment (top used headlines):**")
    top_items = per_item[:min(len(per_item), 12)]
    for x in top_items:
        lab = x["label"]
        conf = x["conf"]
        pill_cls = "good" if "Positive" in lab else "bad" if "Negative" in lab else "neutral"
        st.markdown(f'<span class="pill {pill_cls}">{lab} ({conf:.0%})</span> {x["text"][:140]}{"..." if len(x["text"])>140 else ""}',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
st.caption("⚠️ Educational demo only. Not financial advice. Markets can move for reasons not captured by headlines or USD/INR.")
