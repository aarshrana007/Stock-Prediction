# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-15T15:18:23.060930Z","iopub.execute_input":"2025-12-15T15:18:23.061098Z","iopub.status.idle":"2025-12-15T15:19:21.329718Z","shell.execute_reply.started":"2025-12-15T15:18:23.061073Z","shell.execute_reply":"2025-12-15T15:19:21.328901Z"}}
# CELL 1: Imports + GitHub Recent News Fetcher (Your Code Modified)
import pandas as pd
import requests
from io import StringIO
from kaggle_secrets import UserSecretsClient
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load historical data (old CSV)
print("Loading historical news data...")
df_hist = pd.read_csv("/kaggle/input/indian-financial-news-articles-20032020/IndianFinancialNews.csv")
df_hist = df_hist.dropna(subset=['Title']).copy()
df_hist['Description'] = df_hist['Title'].astype(str)
df_hist['date'] = pd.to_datetime(df_hist['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
print(f"Historical data loaded: {len(df_hist)} rows")

# Create historical embeddings
print("Creating historical embeddings...")
hist_embeddings = embedder.encode(df_hist['Description'].tolist(), batch_size=64, show_progress_bar=True)
hist_embeddings = np.array(hist_embeddings).astype('float32')

# %% [code] {"execution":{"iopub.status.busy":"2025-12-15T15:33:13.185861Z","iopub.execute_input":"2025-12-15T15:33:13.186428Z","iopub.status.idle":"2025-12-15T15:33:13.264313Z","shell.execute_reply.started":"2025-12-15T15:33:13.186403Z","shell.execute_reply":"2025-12-15T15:33:13.263725Z"}}
 # Get GitHub token from Kaggle secrets
    user_secrets = UserSecretsClient()
    token = user_secrets.get_secret("GITHUB_TOKEN")
    if not token:
        raise ValueError("GitHub token not found! Add it in Kaggle Secrets with key 'GITHUB_TOKEN'.")
    
    headers = {'Authorization': f'token {token}'}
    
    # GitHub API URL to list files in the folder
    folder_url = f"https://api.github.com/repos/{'aarshrana007'}/{'Stock_Analysis'}/contents/{'news_data'}"  
    

# %% [code] {"execution":{"iopub.status.busy":"2025-12-15T15:48:52.306386Z","iopub.execute_input":"2025-12-15T15:48:52.307107Z","iopub.status.idle":"2025-12-15T15:48:54.029324Z","shell.execute_reply.started":"2025-12-15T15:48:52.307082Z","shell.execute_reply":"2025-12-15T15:48:54.028575Z"}}
# CELL 2: Fixed GitHub Fetcher with Robust Date Parsing
def load_recent_news_from_github():
    repo_owner = "aarshrana007"
    repo_name = "Stock_Analysis"
    folder_path = "news_data"
    
    try:
        user_secrets = UserSecretsClient()
        token = user_secrets.get_secret("GITHUB_TOKEN")
    except:
        token = None
    
    headers = {'Authorization': f'token {token}'} if token else {}
    
    folder_url = "https://api.github.com/repos/aarshrana007/Stock_Analysis/contents/news_data"
    
    try:
        response = requests.get(folder_url, headers=headers)
        if response.status_code != 200:
            print("GitHub folder not accessible.")
            return pd.DataFrame()
        
        file_list = response.json()
        dfs = []
        total_items = 0
        for file in file_list:
            if file['name'].endswith('.csv'):
                print(f"Loading: {file['name']}")
                csv_data = requests.get(file['download_url']).text
                df = pd.read_csv(StringIO(csv_data))
                if not df.empty:
                    # Combine title + summary
                    df['Description'] = (df.get('title', '') + " " + df.get('summary', '')).str.strip()
                    df = df[df['Description'].str.len() > 20]
                    if not df.empty:
                        dfs.append(df)
                        total_items += len(df)
        
        if dfs:
            recent_df = pd.concat(dfs, ignore_index=True)
            # ROBUST DATE PARSING - handles any format
            recent_df['date'] = pd.to_datetime(recent_df['published'], errors='coerce', infer_datetime_format=True, utc=True)
            recent_df['date'] = recent_df['date'].dt.strftime('%Y-%m-%d')
            recent_df = recent_df.dropna(subset=['date'])  # Drop invalid dates
            print(f"Successfully loaded {total_items} recent news items from {len(dfs)} files!")
            return recent_df[['Description', 'title', 'link', 'published', 'date', 'source']]
        else:
            print("No valid news found.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

recent_news_df = load_recent_news_from_github()
if not recent_news_df.empty:
    display(recent_news_df.head())
    print(f"Total recent news: {len(recent_news_df)}")
else:
    print("No recent data loaded")

# %% [code] {"execution":{"iopub.status.busy":"2025-12-15T15:49:00.603498Z","iopub.execute_input":"2025-12-15T15:49:00.604231Z","iopub.status.idle":"2025-12-15T15:49:00.614460Z","shell.execute_reply.started":"2025-12-15T15:49:00.604179Z","shell.execute_reply":"2025-12-15T15:49:00.613571Z"}}
recent_news_df

# %% [code] {"execution":{"iopub.status.busy":"2025-12-15T15:49:04.453372Z","iopub.execute_input":"2025-12-15T15:49:04.453956Z","iopub.status.idle":"2025-12-15T15:49:07.109060Z","shell.execute_reply.started":"2025-12-15T15:49:04.453933Z","shell.execute_reply":"2025-12-15T15:49:07.108320Z"}}
# CELL 3: Real-Time Market + Top 10 News + Interactive Predictor
from IPython.display import display, clear_output
import ipywidgets as widgets

# Real-Time Market Snapshot
print("\n📊 Real-Time Market Snapshot")
try:
    nifty = yf.Ticker("^NSEI").info
    sensex = yf.Ticker("^BSESN").info
    n_price = nifty.get('regularMarketPrice', 'N/A')
    n_change = nifty.get('regularMarketChangePercent', 0)
    s_price = sensex.get('regularMarketPrice', 'N/A')
    s_change = sensex.get('regularMarketChangePercent', 0)
    print(f"Nifty 50: {n_price} ({n_change:+.2f}%)")
    print(f"Sensex: {s_price} ({s_change:+.2f}%)")
except Exception as e:
    print("Market data unavailable:", e)

# Top 10 Latest News from GitHub
print("\n📰 Top 10 Latest Financial News (Your GitHub Source)")
if not recent_news_df.empty:
    latest = recent_news_df.sort_values(by='date', ascending=False).head(10)
    for i, row in latest.iterrows():
        print(f"{i+1}. {row['Description']}")
else:
    print("No recent news available")

# Interactive Predictor
print("\n🔮 Stock Predictor")
txt = widgets.Text(value="RELIANCE", description="Stock:", placeholder="SBI, TCS...")
btn = widgets.Button(description="Predict", button_style="success")
out = widgets.Output()

def predict_stock(b):
    with out:
        clear_output()
        stock = txt.value.strip().upper()
        if not stock:
            print("Enter a stock symbol")
            return
        
        ticker = f"{stock}.NS"
        print(f"\nAnalyzing {stock}...")
        
        # Use recent news for sentiment
        if not recent_news_df.empty:
            news_texts = recent_news_df['Description'].tolist()
        else:
            news_texts = df_combined['Description'].sample(20).tolist()
        
        scores = [sia.polarity_scores(t)['compound'] for t in news_texts]
        sentiment = np.mean(scores)
        print(f"Sentiment: {sentiment:+.3f} → {'Bullish' if sentiment>0.05 else 'Bearish' if sentiment<-0.05 else 'Neutral'}")
        
        query_vec = embedder.encode([" ".join(news_texts[:50])]).astype('float32')
        sim = np.dot(embeddings, query_vec.T).flatten() / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-8)
        top_idx = np.argsort(sim)[-10:][::-1]
        
        changes = []
        for idx in top_idx:
            row = df_combined.iloc[idx]
            try:
                data = yf.download(ticker, start=row['date'], 
                                 end=(pd.to_datetime(row['date']) + timedelta(days=15)).strftime('%Y-%m-%d'), 
                                 progress=False)
                if len(data) >= 4:
                    pct = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                    changes.append(pct)
            except:
                continue
        
        final_move = np.mean(changes) + sentiment * 10 if changes else sentiment * 12
        confidence = int(68 + abs(sentiment)*25)
        direction = "UP 🟢" if final_move > 0 else "DOWN 🔴"
        
        print(f"\nFINAL PREDICTION → {stock}")
        print(f"Expected 5-day move: {direction} {abs(final_move):.1f}%")
        print(f"Confidence: {confidence}%")

btn.on_click(predict_stock)
display(widgets.VBox([txt, btn, out]))

# Quick test
print("\nQuick test with RELIANCE...")
predict_stock(None)

# %% [code]
