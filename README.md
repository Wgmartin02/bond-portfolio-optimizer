# bond-portfolio-optimizer# Bond Portfolio Optimizer

This project implements an AI-powered bond portfolio optimization tool that allows users to select up to 4 bond ETFs from a list of the top 50 by AUM. The application optimizes the portfolio allocation to maximize risk-adjusted returns using the Sharpe ratio.

## Features

- Select from 50 popular bond ETFs
- Portfolio optimization using the Sharpe ratio
- Monte Carlo simulation to visualize the efficient frontier
- Interactive visualizations of portfolio composition and risk metrics
- Correlation analysis between selected ETFs
- AI-powered risk assessment

## Installation

1. Clone this repository or download the code files
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:

```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)
3. Use the sidebar to select 2-4 bond ETFs from the list
4. Adjust the risk-free rate and historical period as needed
5. View the optimized portfolio allocation and risk metrics

## How It Works

1. **ETF Data Collection**: The application fetches historical price data for the selected ETFs using the yfinance API
2. **Returns Calculation**: Daily returns are calculated and used to estimate annualized returns and volatility
3. **Monte Carlo Simulation**: Random portfolio weights are generated to visualize the efficient frontier
4. **Portfolio Optimization**: The Sharpe ratio is maximized to find the optimal allocation of assets
5. **Risk Assessment**: AI-powered risk metrics are calculated based on volatility, correlation, and interest rate sensitivity
6. **Visualization**: Interactive charts and tables display the results for easy interpretation

## Project Structure

- `app.py`: Main application file containing all the code
- `requirements.txt`: List of required Python packages
- `README.md`: This file

## Dependencies

- pandas
- numpy
- yfinance
- matplotlib
- streamlit
- scipy
- plotly

## Note

This application is designed for educational purposes as a final project for a data analytics class. It uses free, publicly available APIs and data sources.
