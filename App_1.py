import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import re
from datetime import datetime, timedelta
from io import StringIO

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


# =========================
# Page config + Styling
# =========================
st.set_page_config(page_title="Indian Stock Signal (FinBERT)", page_icon="📈", layout="centered")

st.markdown("""
<style>
    .big-font {font-size:46px !important; font-weight:bold; text-align:center;}
    .pred-up {background-color:#d4edda; padding:22px; border-radius:14px; border:3px solid #28a745; text-align:center;}
    .pred-down {background-color:#f8d7da; padding:22px; border-radius:14px; border:3px solid #dc3545; text-align:center;}
    .pred-hold {background-color:#fff3cd; padding:22px; border-radius:14px; border:3px solid #ffc107; text-align:center;}
    .news-item {margin-bottom:10px; padding:12px; background-color:#f0f2f6; border-radius:10px;}
    .market-box {padding:14px; border-radius:10px; text-align:center;}
    .small {color:#666; font-size:13px;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Indian Stock Signal App (FinBERT)")
st.markdown("Real-time market snapshot + news sentiment + historical pattern matching. **For learning only, not financial advice.**")


# =========================
# Utilities: cleaning + dates
# =========================
DATE_COL_CANDIDATES = ["date", "Date", "published", "published_ist", "publishedAt", "time", "datetime"]
TEXT_COL_CANDIDATES = ["Description", "Title", "title", "summary", "content", "headline"]

def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)

def extract_date_any(row: pd.Series) -> pd.Timestamp | None:
    """
    Try multiple columns to get a datetime. Returns pd.Timestamp or None.
    """
    for c in DATE_COL_CANDIDATES:
        if c in row.index:
            v = row.get(c)
            if pd.isna(v):
                continue
            s = str(v).strip()
            if not s or s.lower() == "unknown":
                continue
            dt = pd.to_datetime(s, errors="coerce")
            if pd.notna(dt):
                return dt
            # fallback regex YYYY-MM-DD
            m = re.search(r"\d{4}-\d{2}-\d{2}", s)
            if m:
                dt2 = pd.to_datetime(m.group(0), errors="coerce")
                if pd.notna(dt2):
                    return dt2
    return None

def clean_news_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize to columns: ['date','title','link','source','Description']
    - robust against missing columns
    - removes empty text, duplicates
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date","title","link","source","Description"])

    df = df.copy()

    # Build Description
    if "Description" not in df.columns:
        # try title+summary or any text column
        title_col = "title" if "title" in df.columns else ("Title" if "Title" in df.columns else None)
        summ_col = "summary" if "summary" in df.columns else ("content" if "content" in df.columns else None)

        if title_col and summ_col:
            df["Description"] = (df[title_col].astype(str) + " " + df[summ_col].astype(str)).str.strip()
        elif title_col:
            df["Description"] = df[title_col].astype(str)
        else:
            # pick first available text-ish column
            pick = None
            for c in TEXT_COL_CANDIDATES:
                if c in df.columns:
                    pick = c
                    break
            df["Description"] = df[pick].astype(str) if pick else ""

    # Create standard columns if missing
    if "title" not in df.columns:
        if "Title" in df.columns:
            df["title"] = df["Title"].astype(str)
        else:
            df["title"] = df["Description"].astype(str).str.slice(0, 120)

    if "link" not in df.columns:
        df["link"] = ""

    if "source" not in df.columns:
        df["source"] = ""

    # Create date column robustly
    if "date" not in df.columns:
        df["date"] = None

    # Fill date values using available columns
    if df["date"].isna().any() or (df["date"].astype(str).str.lower() == "unknown").any():
        parsed_list = []
        for _, row in df.iterrows():
            dt = extract_date_any(row)
            parsed_list.append(dt)
        df["date"] = parsed_list

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Final fallback: today (avoid empty)
    df.loc[df["date"].isna(), "date"] = pd.Timestamp(datetime.now().date())

    # Normalize string date if needed
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)

    # Clean text
    df["Description"] = df["Description"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["Description"].str.len() >= 20].copy()

    # Deduplicate
    df["__dedupe_key"] = (df["Description"].str.lower().str.slice(0, 200) + "||" + df["date"].astype(str))
    df = df.drop_duplicates(subset="__dedupe_key").drop(columns=["__dedupe_key"])

    return df[["date","title","link","source","Description"]]


# =========================
# Model loaders
# =========================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_finbert_pipeline():
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return nlp

embedder = load_embedder()
nlp_sentiment = load_finbert_pipeline()

FINBERT_MAP = {
    "positive": ("Positive 🟢", +1),
    "negative": ("Negative 🔴", -1),
    "neutral":  ("Neutral ⚪", 0),
}


def finbert_score_headlines(texts: list[str], max_items: int = 20):
    """
    Returns:
      overall_score (float): average signed confidence
      label (str): Bullish/Bearish/Neutral
      per_item (list[dict]): [{text, label, conf, signed}]
    """
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    texts = texts[:max_items]
    if not texts:
        return 0.0, "Neutral ⚪", []

    results = nlp_sentiment(texts)

    per_item = []
    signed_sum = 0.0
    for t, r in zip(texts, results):
        raw_label = str(r["label"]).lower().strip()
        conf = float(r["score"])
        pretty, sign = FINBERT_MAP.get(raw_label, ("Neutral ⚪", 0))
        signed = sign * conf
        signed_sum += signed
        per_item.append({"text": t, "raw": raw_label, "label": pretty, "conf": conf, "signed": signed})

    overall = signed_sum / len(per_item)
    mood = "Bullish 🟢" if overall > 0.10 else "Bearish 🔴" if overall < -0.10 else "Neutral ⚪"
    return overall, mood, per_item


# =========================
# Data loading
# =========================
@st.cache_data
def load_historical_news(path="IndianFinancialNews.csv"):
    df = pd.read_csv(path)
    # try common columns
    if "Title" in df.columns:
        df["Description"] = df["Title"].astype(str)
    elif "title" in df.columns:
        df["Description"] = df["title"].astype(str)
    else:
        # fallback: use first column as text (not ideal)
        df["Description"] = df.iloc[:, 0].astype(str)

    # date
    if "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.Timestamp(datetime.now().date())

    df["date"] = df["date"].dt.date.astype(str)
    df["Description"] = df["Description"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.dropna(subset=["Description"]).copy()
    df = df[df["Description"].str.len() >= 20].copy()
    df = df.drop_duplicates(subset=["Description", "date"])

    embeddings = embedder.encode(df["Description"].tolist(), batch_size=64)
    return df[["Description","date"]].reset_index(drop=True), np.array(embeddings).astype("float32")


@st.cache_data(ttl=1800)
def load_recent_news_from_github():
    """
    Reads CSV files from your private GitHub repo folder.
    Expects secrets: GITHUB_TOKEN
    """
    token = st.secrets.get("GITHUB_TOKEN", "")
    if not token:
        return pd.DataFrame()

    headers = {"Authorization": f"token {token}"}
    folder_url = "https://api.github.com/repos/aarshrana007/Stock_Analysis/contents/news_data"

    try:
        resp = requests.get(folder_url, headers=headers, timeout=20)
        if resp.status_code != 200:
            return pd.DataFrame()

        files = resp.json()
        dfs = []
        for f in files:
            if f.get("name", "").endswith(".csv"):
                csv_txt = requests.get(f["download_url"], headers=headers, timeout=20).text
                tmp = pd.read_csv(StringIO(csv_txt))
                if not tmp.empty:
                    dfs.append(tmp)

        if not dfs:
            return pd.DataFrame()

        raw = pd.concat(dfs, ignore_index=True)
        return raw

    except Exception:
        return pd.DataFrame()


def google_news_rss(query: str, max_items: int = 15) -> pd.DataFrame:
    """
    Fetch news via Google News RSS (no heavy scraping).
    """
    # RSS endpoint
    url = "https://news.google.com/rss/search"
    params = {
        "q": query,
        "hl": "en-IN",
        "gl": "IN",
        "ceid": "IN:en"
    }

    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        xml = r.text

        # Minimal XML parsing using pandas read_xml if available
        try:
            items = pd.read_xml(xml, xpath=".//item")
        except Exception:
            # fallback regex parse titles/links (simple)
            titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", xml)
            links  = re.findall(r"<link>(.*?)</link>", xml)
            pubs   = re.findall(r"<pubDate>(.*?)</pubDate>", xml)
            # first <title> is channel title; skip it
            if titles:
                titles = titles[1:]
            items = pd.DataFrame({"title": titles[:max_items], "link": links[:max_items], "pubDate": pubs[:max_items]})

        if items is None or items.empty:
            return pd.DataFrame()

        # Standardize
        out = pd.DataFrame()
        out["title"] = items.get("title", "").astype(str)
        out["link"] = items.get("link", "").astype(str)
        out["source"] = "GoogleNewsRSS"
        out["published"] = items.get("pubDate", items.get("pubdate", items.get("published", "")))

        # Description is just title (RSS doesn't contain full summary reliably)
        out["Description"] = out["title"].astype(str)

        out = clean_news_df(out)
        return out.head(max_items)

    except Exception:
        return pd.DataFrame(columns=["date","title","link","source","Description"])


# =========================
# Stock keyword map (for better matching)
# =========================
NAME_MAP = {
    "AXISBANK": ["axis bank", "axisbank", "axis"],
    "SBIN": ["sbi", "state bank", "state bank of india"],
    "SBI": ["sbi", "state bank", "state bank of india"],
    "RELIANCE": ["reliance", "ril", "reliance industries"],
    "TCS": ["tcs", "tata consultancy", "tata consultancy services"],
    "INFY": ["infosys", "infy"],
    "HDFCBANK": ["hdfc bank", "hdfcbank"],
    "ICICIBANK": ["icici bank", "icicibank"],
}

def build_keywords(symbol: str) -> list[str]:
    symbol = symbol.upper().strip()
    kws = [symbol]
    if symbol in NAME_MAP:
        kws += NAME_MAP[symbol]
    return list(dict.fromkeys(kws))

def filter_news_for_stock(news_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if news_df is None or news_df.empty:
        return pd.DataFrame(columns=["date","title","link","source","Description"])
    kws = build_keywords(symbol)
    patt = "|".join([re.escape(k) for k in kws])
    mask = news_df["Description"].str.contains(patt, case=False, na=False) | news_df["title"].astype(str).str.contains(patt, case=False, na=False)
    out = news_df.loc[mask].copy()
    out = out.sort_values("date", ascending=False)
    return out


# =========================
# Market snapshot
# =========================
def market_snapshot():
    def get_info(ticker):
        try:
            return yf.Ticker(ticker).info
        except Exception:
            return {}

    nifty = get_info("^NSEI")
    sensex = get_info("^BSESN")

    # FX
    usd_inr = None
    usd_change = 0.0
    try:
        fx = yf.download("USDINR=X", period="10d", interval="1d", progress=False)
        if not fx.empty:
            usd_inr = float(fx["Close"].iloc[-1])
            usd_change = float((fx["Close"].iloc[-1] / fx["Close"].iloc[-6] - 1) * 100) if len(fx) >= 6 else float((fx["Close"].iloc[-1] / fx["Close"].iloc[0] - 1) * 100)
    except Exception:
        usd_inr = None
        usd_change = 0.0

    return {
        "nifty_price": nifty.get("regularMarketPrice", "N/A"),
        "nifty_change": float(nifty.get("regularMarketChangePercent", 0.0) or 0.0),
        "sensex_price": sensex.get("regularMarketPrice", "N/A"),
        "sensex_change": float(sensex.get("regularMarketChangePercent", 0.0) or 0.0),
        "usd_inr": usd_inr,
        "usd_inr_5d_change_pct": usd_change
    }


# =========================
# Main app UI
# =========================
# Load historical + recent
df_hist, hist_embeddings = load_historical_news()
raw_recent = load_recent_news_from_github()
recent_news_df = clean_news_df(raw_recent) if not raw_recent.empty else pd.DataFrame(columns=["date","title","link","source","Description"])

# Combine embeddings
if not recent_news_df.empty:
    recent_emb = embedder.encode(recent_news_df["Description"].tolist(), batch_size=64)
    recent_emb = np.array(recent_emb).astype("float32")
    embeddings = np.vstack([hist_embeddings, recent_emb])
    df_combined = pd.concat([df_hist, recent_news_df[["Description","date"]]], ignore_index=True)
    st.success(f"Loaded {len(recent_news_df)} recent news items from your private repo.")
else:
    embeddings = hist_embeddings
    df_combined = df_hist
    st.info("No recent GitHub news found. Using historical dataset + Google News RSS fallback when needed.")


# Market Snapshot
st.subheader("📊 Market Snapshot")
snap = market_snapshot()
c1, c2, c3 = st.columns(3)

with c1:
    color = "#d4edda" if snap["nifty_change"] > 0 else "#f8d7da"
    st.markdown(
        f"<div class='market-box' style='background-color:{color}'>"
        f"<h4>Nifty 50</h4><h3>{snap['nifty_price']}</h3><p>{snap['nifty_change']:+.2f}%</p></div>",
        unsafe_allow_html=True
    )

with c2:
    color = "#d4edda" if snap["sensex_change"] > 0 else "#f8d7da"
    st.markdown(
        f"<div class='market-box' style='background-color:{color}'>"
        f"<h4>Sensex</h4><h3>{snap['sensex_price']}</h3><p>{snap['sensex_change']:+.2f}%</p></div>",
        unsafe_allow_html=True
    )

with c3:
    fx_color = "#f8d7da" if (snap["usd_inr_5d_change_pct"] > 0) else "#d4edda"
    usd_inr_val = "N/A" if snap["usd_inr"] is None else f"{snap['usd_inr']:.2f}"
    st.markdown(
        f"<div class='market-box' style='background-color:{fx_color}'>"
        f"<h4>USD/INR</h4><h3>{usd_inr_val}</h3><p>5D {snap['usd_inr_5d_change_pct']:+.2f}%</p>"
        f"<div class='small'>USD/INR ↑ means INR weaker</div></div>",
        unsafe_allow_html=True
    )

st.markdown("---")

# Latest news section
st.subheader("📰 Latest News (From Your Inputs)")
if not recent_news_df.empty:
    latest = recent_news_df.sort_values("date", ascending=False).head(10)
    for _, row in latest.iterrows():
        st.markdown(
            f"<div class='news-item'><strong>{row['date']}</strong> — {row['Description']}</div>",
            unsafe_allow_html=True
        )
else:
    st.info("No recent news in your input files.")

st.markdown("---")


# =========================
# Predictor
# =========================
st.subheader("🔮 Stock Predictor (5-Day Move)")
stock = st.text_input("Enter NSE Stock Symbol", value="AXISBANK").upper().strip()

# Optional: bank sensitivity toggle for USD/INR impact
is_bank = st.checkbox("Treat this as a BANK stock (apply USD/INR rule)", value=True)

if st.button("Predict", type="primary"):
    ticker = f"{stock}.NS"

    with st.spinner("Working..."):
        # 1) Get stock-specific news from inputs
        stock_news = filter_news_for_stock(recent_news_df, stock)

        # 2) If no news, fetch Google RSS fallback
        if stock_news.empty:
            q = " OR ".join(build_keywords(stock)) + " stock"
            rss = google_news_rss(q, max_items=15)
            stock_news = rss

        # 3) If still empty, fallback to generic latest combined news snippets
        if stock_news.empty:
            st.warning("No stock-specific news found. Falling back to generic news sentiment.")
            pool_texts = df_combined["Description"].sample(min(30, len(df_combined))).tolist()
        else:
            pool_texts = stock_news["Description"].tolist()

        # 4) FinBERT sentiment + per-headline sentiments
        sent_score, sent_label, sent_items = finbert_score_headlines(pool_texts, max_items=15)
        st.write(f"**FinBERT Sentiment:** {sent_label} ({sent_score:+.3f})")

        # Show the actual news that influenced decision (latest few)
        st.markdown("### 🧾 Latest news used (with sentiment)")
        if stock_news.empty:
            st.info("Using generic news pool (no stock-specific news found).")
        else:
            shown = stock_news.sort_values("date", ascending=False).head(10)
            for _, row in shown.iterrows():
                st.markdown(
                    f"- **{row['date']}** — {row['title'][:120]} "
                    + (f" ([link]({row['link']}))" if row['link'] else "")
                )

        # Also show FinBERT label on the top sentiment items
        st.markdown("### 🧠 FinBERT headline sentiments (top items)")
        for i, item in enumerate(sent_items[:10], start=1):
            st.write(f"{i}. {item['label']} ({item['conf']:.1%}) — {item['text'][:160]}")

        # 5) Similarity search (historical pattern matching)
        query_vec = embedder.encode([" ".join(pool_texts[:50])]).astype("float32")
        denom = (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-8)
        sim = (np.dot(embeddings, query_vec.T).flatten() / denom)

        top_idx = np.argsort(sim)[-10:][::-1]

        # 6) For each similar historical news date, check how stock moved next ~15 days
        changes = []
        st.markdown("### 📌 Similar historical events (and stock move after)")
        for idx in top_idx:
            row = df_combined.iloc[idx]
            dt0 = pd.to_datetime(row["date"], errors="coerce")
            if pd.isna(dt0):
                continue
            start = dt0.strftime("%Y-%m-%d")
            end = (dt0 + timedelta(days=15)).strftime("%Y-%m-%d")

            try:
                data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, timeout=15)
                if len(data) >= 4:
                    pct = (data["Close"].iloc[-1] / data["Close"].iloc[0] - 1) * 100
                    changes.append(pct)
                    st.write(f"**{row['date']}** → **{pct:+.1f}%** | {row['Description'][:120]}...")
            except Exception:
                continue

        # 7) USD/INR adjustment (simple business rule)
        # If USD/INR rises (INR weakens), often negative for banks (risk-off, flows, inflation)
        fx_adj = 0.0
        usd_inr_5d = float(snap["usd_inr_5d_change_pct"])
        if is_bank:
            # If USD/INR up => bearish => subtract
            fx_adj = -0.35 * usd_inr_5d  # tune this weight
        else:
            fx_adj = 0.0

        # 8) Final move calculation (transparent)
        base_move = float(np.mean(changes)) if changes else 0.0
        sentiment_adj = float(sent_score * 8.0)  # tune weight
        final_move = base_move + sentiment_adj + fx_adj

        # Confidence heuristic (simple)
        conf = 65
        conf += int(min(20, abs(sent_score) * 80))
        conf += 5 if len(changes) >= 5 else 0
        conf = int(min(92, max(55, conf)))

        # 9) Convert to Buy/Sell/Hold (simple signal)
        # (Not advice - just a demo)
        if final_move > 1.0:
            signal = "BUY"
            box = "pred-up"
            direction = "UP 🟢"
        elif final_move < -1.0:
            signal = "SELL"
            box = "pred-down"
            direction = "DOWN 🔴"
        else:
            signal = "HOLD"
            box = "pred-hold"
            direction = "SIDEWAYS 🟡"

        # 10) Print “thinking”
        st.markdown("## 🧩 Model thinking (transparent)")
        st.write(f"- **Base move** (avg of similar historical outcomes): **{base_move:+.2f}%**")
        st.write(f"- **Sentiment adjustment** (FinBERT score × weight): **{sentiment_adj:+.2f}%**")
        if is_bank:
            st.write(f"- **USD/INR adjustment** (bank rule, USD/INR 5D {usd_inr_5d:+.2f}%): **{fx_adj:+.2f}%**")
        st.write(f"= **Final expected 5-day move:** **{final_move:+.2f}%**")
        st.write(f"= **Signal:** **{signal}** | **Confidence:** **{conf}%**")

        # 11) Present the result nicely
        st.markdown(
            f'<div class="{box}">'
            f'<div class="big-font">{signal} — {direction} {abs(final_move):.1f}%</div>'
            f'<h3>{stock} expected 5-day move</h3>'
            f'<p><strong>Confidence: {conf}%</strong></p>'
            f'</div>',
            unsafe_allow_html=True
        )

st.caption("Disclaimer: Educational demo. Not investment advice.")
