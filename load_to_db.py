# load_to_db.py

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# --- 1. SETUP ---
# Load environment variables
load_dotenv()

# Database credentials
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# Create database connection string and engine
db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(db_url)

# File paths for the CSV data
stock_csv_path = "stock_price_data.csv"
news_csv_path = "news_headlines_data.csv"

# Mapping for company names to ticker symbols
ticker_map = {
    "Apple": "AAPL",
    "Google": "GOOGL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Meta": "META"
}


# --- 2. DEFINE TABLE CREATION QUERIES ---
create_tables_sql = """
DROP TABLE IF EXISTS fact_news_articles, fact_stock_prices, dim_tickers, dim_dates, dim_news_sources CASCADE;

CREATE TABLE dim_tickers (
    ticker_id SERIAL PRIMARY KEY,
    ticker_symbol VARCHAR(10) UNIQUE NOT NULL,
    company_name VARCHAR(255)
);

CREATE TABLE dim_dates (
    date_id SERIAL PRIMARY KEY,
    full_date DATE UNIQUE NOT NULL,
    day INT,
    month INT,
    year INT
);

CREATE TABLE dim_news_sources (
    source_id SERIAL PRIMARY KEY,
    source_name VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE fact_stock_prices (
    price_id SERIAL PRIMARY KEY,
    ticker_id INT REFERENCES dim_tickers(ticker_id),
    date_id INT REFERENCES dim_dates(date_id),
    open_price NUMERIC,
    high_price NUMERIC,
    low_price NUMERIC,
    close_price NUMERIC,
    adj_close_price NUMERIC,
    volume BIGINT,
    UNIQUE(ticker_id, date_id)
);

CREATE TABLE fact_news_articles (
    article_id SERIAL PRIMARY KEY,
    ticker_id INT REFERENCES dim_tickers(ticker_id),
    date_id INT REFERENCES dim_dates(date_id),
    source_id INT REFERENCES dim_news_sources(source_id),
    title TEXT,
    description TEXT,
    url TEXT UNIQUE
);
"""


# --- 3. ETL PROCESS ---
def run_etl():
    with engine.connect() as conn:
        print("Connection to database established.")
        
        # Start a transaction
        with conn.begin() as transaction:
            try:
                # Step 1: Create tables
                print("Creating database tables...")
                conn.execute(text(create_tables_sql))
                print("Tables created successfully.")
                
                # --- Step 2: Extract and Transform Data ---
                stock_df = pd.read_csv(stock_csv_path)
                # stock_df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
                news_df = pd.read_csv(news_csv_path)
                news_df.drop_duplicates(subset=['url'], inplace=True)

                # Standardize date columns
                stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date
                news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt']).dt.date

                # --- Step 3: Load Dimension Tables ---
                
                # dim_tickers
                dim_tickers_df = pd.DataFrame(ticker_map.items(), columns=['company_name', 'ticker_symbol'])
                dim_tickers_df.to_sql('dim_tickers', conn, if_exists='append', index=False)
                print("Loaded dim_tickers.")

                # dim_dates
                all_dates = pd.concat([stock_df['Date'], news_df['publishedAt']]).unique()
                dim_dates_df = pd.DataFrame({'full_date': all_dates})
                dim_dates_df['day'] = pd.to_datetime(dim_dates_df['full_date']).dt.day
                dim_dates_df['month'] = pd.to_datetime(dim_dates_df['full_date']).dt.month
                dim_dates_df['year'] = pd.to_datetime(dim_dates_df['full_date']).dt.year
                dim_dates_df.to_sql('dim_dates', conn, if_exists='append', index=False)
                print("Loaded dim_dates.")
                
                # dim_news_sources
                dim_sources_df = pd.DataFrame({'source_name': news_df['source'].apply(lambda x: eval(x)['name']).unique()})
                dim_sources_df.to_sql('dim_news_sources', conn, if_exists='append', index=False)
                print("Loaded dim_news_sources.")
                
                # --- Step 4: Prepare and Load Fact Tables ---
                
                # Fetch dimension data back from DB to map names to IDs
                tickers_map_db = pd.read_sql("SELECT ticker_id, ticker_symbol FROM dim_tickers", conn).set_index('ticker_symbol')
                dates_map_db = pd.read_sql("SELECT date_id, full_date FROM dim_dates", conn).set_index('full_date')
                sources_map_db = pd.read_sql("SELECT source_id, source_name FROM dim_news_sources", conn).set_index('source_name')

                # fact_stock_prices
                fact_stock_df = stock_df.copy()
                fact_stock_df['ticker_id'] = fact_stock_df['Ticker'].map(tickers_map_db['ticker_id'])
                fact_stock_df['date_id'] = fact_stock_df['Date'].map(dates_map_db['date_id'])
                fact_stock_df = fact_stock_df[['ticker_id', 'date_id', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
                fact_stock_df.columns = ['ticker_id', 'date_id', 'open_price', 'high_price', 'low_price', 'close_price', 'adj_close_price', 'volume']
                fact_stock_df.to_sql('fact_stock_prices', conn, if_exists='append', index=False)
                print("Loaded fact_stock_prices.")

                # fact_news_articles
                fact_news_df = news_df.copy()
                fact_news_df['ticker_symbol'] = fact_news_df['TickerQuery'].map(ticker_map)
                fact_news_df['ticker_id'] = fact_news_df['ticker_symbol'].map(tickers_map_db['ticker_id'])
                fact_news_df['date_id'] = fact_news_df['publishedAt'].map(dates_map_db['date_id'])
                fact_news_df['source_name'] = news_df['source'].apply(lambda x: eval(x)['name'])
                fact_news_df['source_id'] = fact_news_df['source_name'].map(sources_map_db['source_id'])
                fact_news_df = fact_news_df[['ticker_id', 'date_id', 'source_id', 'title', 'description', 'url']].dropna()
                fact_news_df.to_sql('fact_news_articles', conn, if_exists='append', index=False)
                print("Loaded fact_news_articles.")

                transaction.commit()
                print("\nETL process completed successfully!")
            except Exception as e:
                print(f"An error occurred: {e}")
                transaction.rollback()
                print("Transaction rolled back.")


if __name__ == "__main__":
    run_etl()