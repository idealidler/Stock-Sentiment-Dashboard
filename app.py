# app.py

import streamlit as st
import pandas as pd
import joblib
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIG ---
st.set_page_config(page_title="Stock Predictor", layout="wide")


# --- DATABASE CONNECTION & DATA LOADING ---
# Use Streamlit's caching to avoid reloading data on every interaction
@st.cache_resource
def get_db_engine():
    """Returns a SQLAlchemy engine connected to the database."""
    load_dotenv()
    db_url = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(db_url)

@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_data(_engine):
    """Loads data from the database using the provided engine."""
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

# --- FEATURE ENGINEERING ---
def engineer_features(df):
    """Creates lagged and moving average features."""
    df_featured = df.copy()
    lags = [1, 2, 5, 10]
    for lag in lags:
        df_featured[f'price_change_lag_{lag}'] = df_featured.groupby('ticker_symbol')['adj_close_price'].pct_change(periods=lag)
        df_featured[f'news_count_lag_{lag}'] = df_featured.groupby('ticker_symbol')['news_count'].shift(lag)
    df_featured['ma_5'] = df_featured.groupby('ticker_symbol')['adj_close_price'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
    df_featured['ma_10'] = df_featured.groupby('ticker_symbol')['adj_close_price'].rolling(window=10, min_periods=1).mean().reset_index(0, drop=True)
    return df_featured.dropna()

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """Loads the pre-trained RandomForest model."""
    try:
        model = joblib.load('random_forest_stock_predictor.joblib')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'random_forest_stock_predictor.joblib' is in the same directory.")
        return None

# --- MAIN APP LOGIC ---
st.title("📈 Simple Stock Price Predictor")
st.write("This dashboard uses a RandomForest model to predict if a stock's price will go up or down the next day.")

# Load data and model
engine = get_db_engine()
full_df = load_data(engine)
model = load_model()

if model is not None and not full_df.empty:
    # Sidebar for user input
    st.sidebar.header("Select Stock")
    ticker_list = sorted(full_df['ticker_symbol'].unique())
    selected_ticker = st.sidebar.selectbox("Choose a stock ticker:", ticker_list)

    # Filter data for the selected ticker
    ticker_df = full_df[full_df['ticker_symbol'] == selected_ticker].copy()
    
    # Engineer features for the selected ticker
    featured_df = engineer_features(ticker_df)

    # Display chart
    st.subheader(f"Adjusted Close Price for {selected_ticker}")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='full_date', y='adj_close_price', data=ticker_df, ax=ax)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    st.pyplot(fig)

    # --- PREDICTION ---
    if not featured_df.empty:
        # Get the most recent data point for prediction
        latest_data = featured_df.tail(1)

        features_for_prediction = [
            'news_count', 'volume',
            'price_change_lag_1', 'price_change_lag_2', 'price_change_lag_5', 'price_change_lag_10',
            'news_count_lag_1', 'news_count_lag_2', 'news_count_lag_5', 'news_count_lag_10',
            'ma_5', 'ma_10'
        ]
        X_latest = latest_data[features_for_prediction]

        # Make prediction
        prediction = model.predict(X_latest)[0]
        prediction_proba = model.predict_proba(X_latest)[0]

        st.subheader("Next Day Price Movement Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.metric(label=f"Model Prediction for {selected_ticker}", value="UP ▲", delta="Positive Outlook")
            else:
                st.metric(label=f"Model Prediction for {selected_ticker}", value="DOWN ▼", delta="Negative Outlook", delta_color="inverse")
        
        with col2:
            st.metric(label="Prediction Confidence", value=f"{max(prediction_proba)*100:.2f}%")

        st.info("Disclaimer: This is an educational model and should not be used for financial decisions.", icon="⚠️")
    else:
        st.warning("Not enough data to make a prediction for the selected stock.")
else:
    st.error("Could not load data or model. Please check the console for errors.")