# Bond Portfolio Optimizer

A simple bond portfolio optimization tool that allows users to select up to 4 bond ETFs from a list of the top 50 by AUM. The application optimizes the portfolio allocation to maximize risk-adjusted returns using the Sharpe ratio.

## Features

- Select from 50 popular bond ETFs
- Portfolio optimization using the Sharpe ratio
- Monte Carlo simulation to visualize the efficient frontier
- Interactive visualizations of portfolio composition and risk metrics
- Correlation analysis between selected ETFs
- AI-powered risk assessment

## Installation

1. Clone this repository or download the code
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Get a free API key from Alpha Vantage:
   - Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key) and sign up for a free API key
   - Replace `YOUR_API_KEY` in the `app.py` file with your actual API key or enter it in the app's sidebar

## Usage

1. Run the Streamlit application:

```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)
3. Enter your Alpha Vantage API key in the sidebar (if you didn't add it to the code)
4. Select 2-4 bond ETFs from the list
5. Adjust the risk-free rate and historical period as needed
6. Click "Run Portfolio Optimization"
7. View the optimized portfolio allocation and risk metrics

## How It Works

1. **ETF Data Collection**: The application fetches historical price data for the selected ETFs using the Alpha Vantage API
2. **Returns Calculation**: Daily returns are calculated and used to estimate annualized returns and volatility
3. **Monte Carlo Simulation**: Random portfolio weights are generated to visualize the efficient frontier
4. **Portfolio Optimization**: The Sharpe ratio is maximized to find the optimal allocation of assets
5. **Risk Assessment**: AI-powered risk metrics are calculated based on volatility, correlation, and interest rate sensitivity
6. **Visualization**: Interactive charts and tables display the results for easy interpretation

## API Limits

The Alpha Vantage free tier has the following limits:
- 25 API requests per day
- 5 API calls per minute

This application is designed to work within these limits by:
- Adding appropriate delays between API calls
- Limiting selections to 4 ETFs maximum
- Providing clear error messages if limits are exceeded

## Note

This application is designed for educational purposes as a final project for a data analytics class. It uses a free, publicly available API for financial data.
