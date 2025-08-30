import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv('NEWS_API_KEY')
if not api_key:
    raise ValueError("NEWS_API_KEY not found. Please set it in your .env file.")

# Define the company names/keywords for our tickers
search_queries = ["Apple", "Google", "Microsoft", "Amazon", "Meta"]

# Set the time frame (e.g., last 30 days - free tier limit)
to_date = datetime.now().date()
from_date = to_date - timedelta(days=29)

all_articles = []

# Fetch news for each query
for query in search_queries:
    print(f"Fetching news for: {query}...")
    url = (
        'https://newsapi.org/v2/everything?'
        f'q="{query}"&' # Use quotes for exact phrase matching
        f'from={from_date}&'
        f'to={to_date}&'
        'language=en&'
        'sortBy=publishedAt&'
        f'apiKey={api_key}'
    )
    
    response = requests.get(url)
    
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        for article in articles:
            # Add the query to know which company the news is about
            article['query'] = query 
        all_articles.extend(articles)
        print(f"  Found {len(articles)} articles.")
    else:
        print(f"  Failed to fetch news for {query}. Status code: {response.status_code}")
        print(f"  Response: {response.text}")


# Convert the list of articles to a pandas DataFrame
news_df = pd.DataFrame(all_articles)

# Select and rename columns for clarity
if not news_df.empty:
    news_df = news_df[['query', 'publishedAt', 'title', 'description', 'source', 'url']]
    news_df.rename(columns={'query': 'TickerQuery'}, inplace=True)
    
    # Save to CSV
    output_path = "news_headlines_data.csv"
    news_df.to_csv(output_path, index=False)
    print(f"\nNews data saved successfully to {output_path}")
else:
    print("\nNo articles found. The DataFrame is empty.")