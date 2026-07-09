# 📊 Portfolio Analysis & Optimization Platform

A Streamlit web app for stock analysis using **CAPM (Capital Asset Pricing Model)** and **Markowitz Mean-Variance Portfolio Optimization**. Pulls live market data, calculates risk/return metrics, and finds optimal portfolio allocations.

## Features

### 📈 CAPM Analysis
- Normalized price comparison across selected stocks vs. a market benchmark
- Beta calculation for each stock
- CAPM expected return vs. actual historical return
- Security Market Line (SML) visualization

### 🧮 Portfolio Optimizer
- Markowitz Mean-Variance Optimization
- Efficient Frontier (with 3,000 simulated random portfolios for context)
- Maximum Sharpe Ratio Portfolio (tangency portfolio)
- Minimum Variance Portfolio
- Capital Allocation Line
- Optimal weight breakdown with pie charts
- Optional short-selling support

### 📐 Risk Metrics
For Max Sharpe, Min Variance, and Equal-Weight portfolios:
- Sharpe Ratio
- Sortino Ratio
- Treynor Ratio
- Beta
- Alpha (Jensen's Alpha)
- Maximum Drawdown
- Value at Risk (95%, 1-day)

## Tech Stack
- **Streamlit** – web app framework
- **yfinance** – historical stock price data
- **pandas / numpy** – data processing
- **scipy** – portfolio optimization (SLSQP)
- **plotly** – interactive charts

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`. (Or, in VS Code, just hit the ▶️ Run button — the script auto-detects and relaunches itself through Streamlit.)

## Usage
1. Enter comma-separated stock tickers in the sidebar (e.g., `AAPL, MSFT, GOOGL, AMZN`)
2. Set a market benchmark (defaults to `^GSPC` for the S&P 500)
3. Adjust the historical data range and risk-free rate
4. Explore the **CAPM Analysis** tab for beta/expected return, and the **Portfolio Optimizer** tab for the efficient frontier and risk metrics

## Notes
- Data is pulled live from Yahoo Finance via `yfinance` — an internet connection is required.
- Long-only weights (0–100%) are the default; enable "Allow short-selling" in the Optimizer tab to relax this constraint.
- This tool is for educational/portfolio purposes and is not financial advice.