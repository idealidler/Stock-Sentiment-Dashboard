# get_stock_data.py (Clean Structured Version)

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Define the stock tickers we're interested in
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]

# Set the date range (last 90 days)
end_date = datetime.now()
start_date = end_date - timedelta(days=90)

print("Fetching data...")

# Download all tickers at once (returns MultiIndex DataFrame)
raw_data = yf.download(
    tickers, 
    start=start_date, 
    end=end_date, 
    group_by="ticker",   # keeps tickers separated in columns
    auto_adjust=False,   # keep both Close and Adj Close
    threads=True
)

# Restructure into a clean long-format DataFrame
all_data = []

for ticker in tickers:
    df = raw_data[ticker].copy()
    df.reset_index(inplace=True)  # move 'Date' back into column
    df["Ticker"] = ticker
    all_data.append(df)

# Concatenate all into one DataFrame
stock_data_df = pd.concat(all_data, ignore_index=True)

# Reorder columns for clarity
stock_data_df = stock_data_df[
    ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
]

# Add daily returns (optional)
stock_data_df["Daily Return"] = stock_data_df.groupby("Ticker")["Adj Close"].pct_change()

# Save to CSV
output_path = "stock_price_data.csv"
stock_data_df.to_csv(output_path, index=False)

print(f"\nStock data saved successfully to {output_path}")
print(f"Shape: {stock_data_df.shape}")
print(stock_data_df.head())
