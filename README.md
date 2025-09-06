# Stock Sentiment & Prediction Dashboard

A full-stack data science project that fetches financial data, performs analysis, and uses a machine learning model with a Large Language Model (LLM) to predict and explain stock price movements.

**Live Demo:** (https://stock-sentiment-dashboard-project.streamlit.app)

---


### ## Features

* **Automated Data Pipeline:** Fetches daily stock prices and news headlines via APIs.
* **ETL & Data Warehousing:** Stores and structures data in a PostgreSQL database using a star schema.
* **ML Prediction:** A Random Forest model predicts whether a stock's price will rise or fall the next day.
* **Interactive Dashboard:** A web-based UI built with Streamlit for easy data visualization and interaction.
* **AI-Powered Insights:** Integrates Google's Gemini API to generate a narrative explanation of the model's prediction based on the day's data.

### ## Tech Stack

* **Backend:** Python
* **Data:** PostgreSQL, Pandas, yfinance, NewsAPI
* **ML:** Scikit-learn, Joblib
* **Frontend/Dashboard:** Streamlit
* **AI:** Google Gemini API
* **Deployment:** Streamlit Community Cloud
* **Version Control:** Git & GitHub

### ## How to Run Locally

1.  Clone the repository: `git clone https://github.com/idealidler/Stock-Sentiment-Dashboard.git`
2.  Create and activate a Python environment.
3.  Install dependencies: `pip install -r requirements.txt`
4.  Set up your PostgreSQL database and `.env` file with your credentials and API keys.
5.  Run the ETL pipeline: `python load_to_db.py`
6.  Launch the app: `streamlit run app.py`

---
