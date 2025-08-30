# check_csv.py

import pandas as pd

try:
    df = pd.read_csv("stock_price_data.csv")
    print("--- First 5 Rows of stock_price_data.csv ---")
    print(df.head())
    print("\n--- Column Names Found ---")
    print(df.columns)
    print("--------------------------")

except FileNotFoundError:
    print("Error: stock_price_data.csv not found.")
except Exception as e:
    print(f"An error occurred: {e}")