# app.py (Final Version with Gemini Integration)

import streamlit as st
import pandas as pd
import joblib
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import seaborn as sns
# --- MODIFIED FOR GEMINI: Import Google's library ---
import google.generativeai as genai

# --- PAGE CONFIG ---
st.set_page_config(page_title="Stock Predictor", layout="wide")

# --- DATABASE CONNECTION & DATA LOADING (No changes here) ---
@st.cache_resource
def get_db_engine():
    load_dotenv()
    db_url = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(db_url)

@st.cache_data(ttl=3600)
def load_data(_engine):
    query = text("""
        SELECT
            d.full_date,
            t.ticker_symbol,
            p.adj_close_price,
            p.volume,
            COALESCE(n.news_count, 0) AS news_count
        FROM fact_stock_prices AS p
        JOIN dim_tickers AS t ON p.ticker_id = t.ticker_id
        JOIN dim_dates AS d ON p.date_id = d.date_id
        LEFT JOIN (
            SELECT ticker_id, date_id, COUNT(article_id) AS news_count
            FROM fact_news_articles
            GROUP BY ticker_id, date_id
        ) AS n ON p.ticker_id = n.ticker_id AND p.date_id = n.date_id
        ORDER BY t.ticker_symbol, d.full_date;
    """)
    with _engine.connect() as connection:
        df = pd.read_sql(query, connection, parse_dates=['full_date'])
    return df

# --- FEATURE ENGINEERING (No changes here) ---
def engineer_features(df):
    df_featured = df.copy()
    lags = [1, 2, 5, 10]
    for lag in lags:
        df_featured[f'price_change_lag_{lag}'] = df_featured.groupby('ticker_symbol')['adj_close_price'].pct_change(periods=lag)
        df_featured[f'news_count_lag_{lag}'] = df_featured.groupby('ticker_symbol')['news_count'].shift(lag)
    df_featured['ma_5'] = df_featured.groupby('ticker_symbol')['adj_close_price'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
    df_featured['ma_10'] = df_featured.groupby('ticker_symbol')['adj_close_price'].rolling(window=10, min_periods=1).mean().reset_index(0, drop=True)
    return df_featured.dropna()

# --- MODEL LOADING (No changes here) ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('random_forest_stock_predictor.joblib')
        return model
    except FileNotFoundError:
        st.error("Model file not found.")
        return None

# --- MODIFIED FOR GEMINI: Updated LLM Integration Function ---
@st.cache_data(ttl=86400) # Cache the response for a day
def get_ai_analysis(ticker, prediction_text, latest_data):
    """Generates a narrative analysis using the Gemini API."""
    # Ensure the API key is loaded
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Gemini API key not found. Please set it in your .env file."
    
    genai.configure(api_key=api_key)

    # Format the latest data for the prompt
    price_change = latest_data['price_change_lag_1'].iloc[0]
    news_count = latest_data['news_count'].iloc[0]
    volume = latest_data['volume'].iloc[0]

    # The prompt is the same as before
    prompt = f"""
    You are a financial analyst providing a brief, easy-to-understand summary.
    
    For the stock {ticker}, our machine learning model predicts the price will move {prediction_text} tomorrow.
    
    This prediction is based on the following key data points from today:
    - The stock's price changed by {price_change:.2%} today.
    - The trading volume was {volume:,.0f} shares.
    - There were {news_count:.0f} news articles related to the company today.

    Based *only* on this data, write a short, one-paragraph narrative explaining this prediction. Do not give financial advice or make definitive statements. Keep the tone neutral and data-focused. Start with "Based on recent data..."
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred with the AI analysis: {e}"


# --- MAIN APP LOGIC (No changes here) ---
st.title("📈 Simple Stock Price Predictor")
st.write("This dashboard uses a RandomForest model to predict if a stock's price will go up or down the next day.")

engine = get_db_engine()
full_df = load_data(engine)
model = load_model()

if model is not None and not full_df.empty:
    st.sidebar.header("Select Stock")
    ticker_list = sorted(full_df['ticker_symbol'].unique())
    selected_ticker = st.sidebar.selectbox("Choose a stock ticker:", ticker_list)

    ticker_df = full_df[full_df['ticker_symbol'] == selected_ticker].copy()
    featured_df = engineer_features(ticker_df)

    st.subheader(f"Adjusted Close Price for {selected_ticker}")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='full_date', y='adj_close_price', data=ticker_df, ax=ax)
    st.pyplot(fig)

    if not featured_df.empty:
        latest_data = featured_df.tail(1)
        features_for_prediction = [
            'news_count', 'volume',
            'price_change_lag_1', 'price_change_lag_2', 'price_change_lag_5', 'price_change_lag_10',
            'news_count_lag_1', 'news_count_lag_2', 'news_count_lag_5', 'news_count_lag_10',
            'ma_5', 'ma_10'
        ]
        X_latest = latest_data[features_for_prediction]

        prediction = model.predict(X_latest)[0]
        prediction_proba = model.predict_proba(X_latest)[0]

        st.subheader("Next Day Price Movement Prediction")
        
        col1, col2 = st.columns(2)
        prediction_text = "UP ▲" if prediction == 1 else "DOWN ▼"
        delta_text = "Positive Outlook" if prediction == 1 else "Negative Outlook"
        
        with col1:
            st.metric(label=f"Model Prediction for {selected_ticker}", value=prediction_text, delta=delta_text, delta_color=("normal" if prediction == 1 else "inverse"))
        
        with col2:
            st.metric(label="Prediction Confidence", value=f"{max(prediction_proba)*100:.2f}%")

        st.subheader("AI Analyst Report")
        if st.button("Generate AI Analysis"):
            with st.spinner("Our AI analyst is crunching the numbers..."):
                analysis = get_ai_analysis(selected_ticker, prediction_text, latest_data)
                st.write(analysis)

        st.info("Disclaimer: This is an educational model and should not be used for financial decisions.", icon="⚠️")
    else:
        st.warning("Not enough data to make a prediction.")
else:
    st.error("Could not load data or model.")