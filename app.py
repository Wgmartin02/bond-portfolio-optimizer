import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import time
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Constants
# Alpha Vantage free API key - replace with your own
ALPHA_VANTAGE_API_KEY = "MH2X46EV62BMGBZ0"  # Get a free API key from https://www.alphavantage.co/support/#api-key

# Top 50 Bond ETFs by AUM
TOP_BOND_ETFS = {
    'AGG': 'iShares Core U.S. Aggregate Bond ETF',
    'BND': 'Vanguard Total Bond Market ETF',
    'VCIT': 'Vanguard Intermediate-Term Corporate Bond ETF',
    'LQD': 'iShares iBoxx $ Investment Grade Corporate Bond ETF',
    'VCSH': 'Vanguard Short-Term Corporate Bond ETF',
    'BSV': 'Vanguard Short-Term Bond ETF',
    'MBB': 'iShares MBS ETF',
    'GOVT': 'iShares U.S. Treasury Bond ETF',
    'VTIP': 'Vanguard Short-Term Inflation-Protected Securities ETF',
    'TIP': 'iShares TIPS Bond ETF',
    'IGIB': 'iShares Intermediate-Term Corporate Bond ETF',
    'MUB': 'iShares National Muni Bond ETF',
    'VGSH': 'Vanguard Short-Term Treasury ETF',
    'VGIT': 'Vanguard Intermediate-Term Treasury ETF',
    'HYG': 'iShares iBoxx $ High Yield Corporate Bond ETF',
    'VMBS': 'Vanguard Mortgage-Backed Securities ETF',
    'IEF': 'iShares 7-10 Year Treasury Bond ETF',
    'SHV': 'iShares Short Treasury Bond ETF',
    'VTEB': 'Vanguard Tax-Exempt Bond ETF',
    'IGSB': 'iShares Short-Term Corporate Bond ETF',
    'SPIB': 'SPDR Portfolio Intermediate Term Corporate Bond ETF',
    'USIG': 'iShares Broad USD Investment Grade Corporate Bond ETF',
    'BIV': 'Vanguard Intermediate-Term Bond ETF',
    'VGLT': 'Vanguard Long-Term Treasury ETF',
    'SPTL': 'SPDR Portfolio Long Term Treasury ETF',
    'TLT': 'iShares 20+ Year Treasury Bond ETF',
    'SPSB': 'SPDR Portfolio Short Term Corporate Bond ETF',
    'JNK': 'SPDR Bloomberg High Yield Bond ETF',
    'SCHR': 'Schwab Intermediate-Term U.S. Treasury ETF',
    'SCHP': 'Schwab U.S. TIPS ETF',
    'SCHO': 'Schwab Short-Term U.S. Treasury ETF',
    'SCHZ': 'Schwab U.S. Aggregate Bond ETF',
    'HYLB': 'Xtrackers USD High Yield Corporate Bond ETF',
    'SHYG': 'iShares 0-5 Year High Yield Corporate Bond ETF',
    'BNDX': 'Vanguard Total International Bond ETF',
    'FLOT': 'iShares Floating Rate Bond ETF',
    'SLQD': 'iShares 0-5 Year Investment Grade Corporate Bond ETF',
    'IGOV': 'iShares International Treasury Bond ETF',
    'EMB': 'iShares J.P. Morgan USD Emerging Markets Bond ETF',
    'BWX': 'SPDR Bloomberg International Treasury Bond ETF',
    'SJNK': 'SPDR Bloomberg Short Term High Yield Bond ETF',
    'VWOB': 'Vanguard Emerging Markets Government Bond ETF',
    'LKOR': 'FlexShares Credit-Scored US Long Corporate Bond Index Fund',
    'GBIL': 'Goldman Sachs Treasury Access 0-1 Year ETF',
    'SUSB': 'iShares ESG Aware 1-5 Year USD Corporate Bond ETF',
    'STIP': 'iShares 0-5 Year TIPS Bond ETF',
    'BKLN': 'Invesco Senior Loan ETF',
    'SHY': 'iShares 1-3 Year Treasury Bond ETF',
    'BLV': 'Vanguard Long-Term Bond ETF',
    'USHY': 'iShares Broad USD High Yield Corporate Bond ETF'
}

def get_etf_data_alpha_vantage(symbol, time_period='1year'):
    """
    Get historical ETF data from Alpha Vantage
    """
    try:
        # Respect the rate limit (5 calls per minute for free API)
        time.sleep(12)  # To ensure we don't exceed rate limits
        
        # Using TIME_SERIES_DAILY_ADJUSTED endpoint which works well for ETFs
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}'
        
        response = requests.get(url)
        data = response.json()
        
        # Check for error messages
        if 'Error Message' in data:
            st.error(f"Error retrieving data for {symbol}: {data['Error Message']}")
            return None
        
        if 'Time Series (Daily)' not in data:
            if 'Note' in data:
                st.warning(f"API limit reached: {data['Note']}")
            else:
                st.error(f"No data available for {symbol}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data['Time Series (Daily)']).T
        
        # Convert columns to numeric
        df = df.astype(float)
        
        # Rename columns for clarity
        df.columns = [col.split('. ')[1] for col in df.columns]
        
        # Sort by date (ascending)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Filter based on time period
        if time_period == '1year':
            start_date = datetime.now() - timedelta(days=365)
            df = df[df.index >= start_date]
        elif time_period == '2year':
            start_date = datetime.now() - timedelta(days=730)
            df = df[df.index >= start_date]
        elif time_period == '3year':
            start_date = datetime.now() - timedelta(days=1095)
            df = df[df.index >= start_date]
        
        # Keep only adjusted close
        return df['adjusted close']
    
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def get_historical_data(tickers, time_period='1year'):
    """
    Get historical price data for multiple tickers
    """
    data = {}
    valid_tickers = []
    
    with st.spinner(f"Fetching data for {len(tickers)} ETFs... This may take a minute."):
        for ticker in tickers:
            ticker_data = get_etf_data_alpha_vantage(ticker, time_period)
            if ticker_data is not None and not ticker_data.empty:
                data[ticker] = ticker_data
                valid_tickers.append(ticker)
    
    if not data:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Check if we have enough data
    if len(df) < 30:  # Need at least 30 data points for meaningful analysis
        st.warning("Not enough historical data points. Results may be less reliable.")
    
    return df

def calculate_returns(prices):
    """
    Calculate daily returns and annualized metrics from price data
    """
    if prices is None or prices.empty:
        return None, None, None
    
    # Calculate daily returns
    daily_returns = prices.pct_change().dropna()
    
    # Calculate annualized returns
    annual_returns = (1 + daily_returns.mean()) ** 252 - 1
    
    # Calculate annualized volatility
    annual_volatility = daily_returns.std() * np.sqrt(252)
    
    return daily_returns, annual_returns, annual_volatility

def calculate_sharpe_ratio(returns, volatility, risk_free_rate=0.03):
    """
    Calculate the Sharpe ratio for a portfolio
    """
    if volatility == 0:
        return 0
    return (returns - risk_free_rate) / volatility

def negative_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.03):
    """
    Calculate the negative Sharpe ratio (for minimization)
    """
    portfolio_return = np.sum(returns * weights)
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Avoid division by zero
    if portfolio_stddev == 0:
        return 0
    
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev
    return -sharpe_ratio

def monte_carlo_simulation(returns, num_portfolios=1000):
    """
    Perform Monte Carlo simulation to generate random portfolio weights
    and calculate their performance metrics
    """
    if returns is None or returns.empty:
        return pd.DataFrame()
    
    num_assets = len(returns.columns)
    results = []
    
    for _ in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        # Calculate portfolio return and volatility
        portfolio_return = np.sum(returns.mean() * 252 * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        
        # Calculate Sharpe ratio
        sharpe_ratio = calculate_sharpe_ratio(portfolio_return, portfolio_volatility)
        
        results.append({
            'weights': weights,
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        })
    
    return pd.DataFrame(results)

def optimize_portfolio(returns, risk_free_rate=0.03):
    """
    Find the optimal portfolio weights to maximize the Sharpe ratio
    """
    if returns is None or returns.empty:
        return None
    
    num_assets = len(returns.columns)
    
    # Initial guess (equal weights)
    initial_guess = np.array([1/num_assets] * num_assets)
    
    # Constraints (weights sum to 1)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds (weights between 0 and 1)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Arguments for the negative Sharpe ratio function
    args = (returns.mean() * 252, returns.cov() * 252, risk_free_rate)
    
    # Optimize
    try:
        result = minimize(
            negative_sharpe_ratio,
            initial_guess,
            args=args,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result['success']:
            return result['x']
        else:
            st.warning("Optimization did not converge. Using equal weights.")
            return initial_guess
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        return initial_guess

def create_efficient_frontier_chart(mc_results, optimal_portfolio, etf_names):
    """
    Create a plotly chart showing the efficient frontier and optimal portfolio
    """
    if mc_results.empty:
        return None
    
    fig = go.Figure()
    
    # Plot random portfolios
    fig.add_trace(go.Scatter(
        x=mc_results['volatility'],
        y=mc_results['return'],
        mode='markers',
        marker=dict(
            size=5,
            color=mc_results['sharpe_ratio'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Sharpe Ratio')
        ),
        name='Random Portfolios'
    ))
    
    # Plot optimal portfolio
    fig.add_trace(go.Scatter(
        x=[optimal_portfolio['volatility']],
        y=[optimal_portfolio['return']],
        mode='markers',
        marker=dict(
            color='red',
            size=12,
            symbol='star'
        ),
        name='Optimal Portfolio'
    ))
    
    fig.update_layout(
        title='Portfolio Efficient Frontier',
        xaxis_title='Annualized Volatility',
        yaxis_title='Annualized Return',
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_portfolio_composition_chart(weights, etf_names):
    """
    Create a plotly pie chart showing portfolio composition
    """
    fig = px.pie(
        values=weights,
        names=etf_names,
        title='Optimal Portfolio Composition'
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500)
    
    return fig

def create_correlation_heatmap(returns, etf_names):
    """
    Create a plotly heatmap showing correlation between assets
    """
    corr_matrix = returns.corr()
    
    fig = px.imshow(
        corr_matrix,
        x=etf_names,
        y=etf_names,
        color_continuous_scale='RdBu_r',
        title='Correlation Matrix'
    )
    
    fig.update_layout(height=500)
    
    return fig

def create_risk_metrics_table(optimal_portfolio, etf_names):
    """
    Create a risk metrics table for the optimal portfolio
    """
    metrics = pd.DataFrame({
        'ETF': etf_names,
        'Weight (%)': [f"{w*100:.2f}%" for w in optimal_portfolio['weights']],
        'Expected Return (%)': [f"{r*100:.2f}%" for r in optimal_portfolio['individual_returns']],
        'Volatility (%)': [f"{v*100:.2f}%" for v in optimal_portfolio['individual_volatilities']]
    })
    
    summary = pd.DataFrame({
        'Metric': ['Portfolio Return', 'Portfolio Volatility', 'Sharpe Ratio'],
        'Value': [
            f"{optimal_portfolio['return']*100:.2f}%",
            f"{optimal_portfolio['volatility']*100:.2f}%",
            f"{optimal_portfolio['sharpe_ratio']:.2f}"
        ]
    })
    
    return metrics, summary

def run_portfolio_optimization():
    st.set_page_config(layout="wide", page_title="Bond Portfolio Optimizer")
    
    st.title("Bond Portfolio Optimizer")
    st.write("""
    This application helps optimize a bond ETF portfolio by finding the best allocation 
    to maximize risk-adjusted returns. Select 2-4 bond ETFs from the list below.
    """)
    
    # API key input
    api_key = st.sidebar.text_input(
        "Alpha Vantage API Key:",
        value=ALPHA_VANTAGE_API_KEY,
        type="password"
    )
    if api_key != ALPHA_VANTAGE_API_KEY:
        global ALPHA_VANTAGE_API_KEY
        ALPHA_VANTAGE_API_KEY = api_key
    
    # Sidebar for ETF selection
    st.sidebar.header("ETF Selection")
    
    # Default ETFs that are most likely to have good data
    default_etfs = ["AGG", "BND"]
    
    selected_etfs = st.sidebar.multiselect(
        "Select 2-4 Bond ETFs:",
        options=list(TOP_BOND_ETFS.keys()),
        format_func=lambda x: f"{x} - {TOP_BOND_ETFS[x]}",
        default=default_etfs,
        max_selections=4
    )
    
    # Risk-free rate input
    risk_free_rate = st.sidebar.slider(
        "Risk-Free Rate (%):",
        min_value=0.0,
        max_value=5.0,
        value=3.0,
        step=0.1
    ) / 100
    
    # Time period selection
    time_period = st.sidebar.selectbox(
        "Historical Period:",
        ["1year", "2year", "3year"],
        index=0
    )
    
    # Check if at least 2 ETFs are selected
    if len(selected_etfs) < 2:
        st.warning("Please select at least 2 ETFs to optimize a portfolio.")
        st.stop()
    
    # Show selected ETFs
    st.subheader("Selected Bond ETFs")
    etf_df = pd.DataFrame([
        {"Ticker": ticker, "Name": TOP_BOND_ETFS[ticker]}
        for ticker in selected_etfs
    ])
    st.table(etf_df)
    
    # Run button
    run_analysis = st.button("Run Portfolio Optimization")
    
    if run_analysis:
        try:
            # Get historical data
            prices = get_historical_data(selected_etfs, time_period)
            
            # Check if we have valid data
            if prices is None or prices.empty:
                st.error("Could not retrieve data for the selected ETFs. Please try different ETFs or make sure your API key is valid.")
                st.stop()
            
            # Verify we have data for all selected ETFs
            if len(prices.columns) < len(selected_etfs):
                missing_etfs = set(selected_etfs) - set(prices.columns)
                st.warning(f"Could not retrieve data for: {', '.join(missing_etfs)}. Proceeding with available ETFs.")
            
            # Check if we have at least 2 ETFs with data
            if len(prices.columns) < 2:
                st.error("Need at least 2 ETFs with data for optimization. Please select different ETFs.")
                st.stop()
            
            # Calculate returns and metrics
            daily_returns, annual_returns, annual_volatility = calculate_returns(prices)
            
            # Run Monte Carlo simulation
            mc_results = monte_carlo_simulation(daily_returns)
            
            # Find optimal portfolio
            optimal_weights = optimize_portfolio(daily_returns, risk_free_rate)
            
            # Calculate metrics for optimal portfolio
            optimal_portfolio = {
                'weights': optimal_weights,
                'return': np.sum(annual_returns * optimal_weights),
                'volatility': np.sqrt(np.dot(optimal_weights.T, np.dot(daily_returns.cov() * 252, optimal_weights))),
                'individual_returns': annual_returns.values,
                'individual_volatilities': annual_volatility.values
            }
            optimal_portfolio['sharpe_ratio'] = calculate_sharpe_ratio(
                optimal_portfolio['return'],
                optimal_portfolio['volatility'],
                risk_free_rate
            )
            
            # Create ETF names list for charts
            etf_names = list(prices.columns)
            
            # Display results
            col1, col2 = st.columns([3, 1])
            
            # Efficient Frontier Chart
            with col1:
                efficient_frontier = create_efficient_frontier_chart(mc_results, optimal_portfolio, etf_names)
                if efficient_frontier:
                    st.plotly_chart(efficient_frontier, use_container_width=True)
            
            # Portfolio Composition
            with col2:
                composition_chart = create_portfolio_composition_chart(optimal_portfolio['weights'], etf_names)
                st.plotly_chart(composition_chart, use_container_width=True)
            
            # Risk Metrics Table
            st.subheader("Portfolio Risk Metrics")
            metrics_df, summary_df = create_risk_metrics_table(optimal_portfolio, etf_names)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.table(metrics_df)
            with col2:
                st.table(summary_df)
            
            # Correlation Heatmap
            st.subheader("ETF Correlation Analysis")
            correlation_chart = create_correlation_heatmap(daily_returns, etf_names)
            st.plotly_chart(correlation_chart, use_container_width=True)
            
            # Historical Performance
            st.subheader("Historical Performance")
            
            # Normalize price data for comparison
            normalized_prices = prices / prices.iloc[0]
            
            fig = px.line(
                normalized_prices,
                title=f"Historical Price Performance ({time_period})",
                labels={"value": "Normalized Price", "variable": "ETF"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add risk assessment results
            st.subheader("AI Risk Assessment")
            
            # Simple risk score based on volatility
            risk_score = int((optimal_portfolio['volatility'] * 100) * 2)
            if risk_score > 100:  # Cap at 100
                risk_score = 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Portfolio Risk Score", f"{risk_score}/100")
                
                if risk_score < 33:
                    st.info("Low Risk: This portfolio has relatively low volatility.")
                elif risk_score < 66:
                    st.warning("Medium Risk: This portfolio has moderate volatility.")
                else:
                    st.error("High Risk: This portfolio has high volatility.")
            
            with col2:
                avg_correlation = daily_returns.corr().values.mean()
                st.metric("Average Correlation", f"{avg_correlation:.2f}")
                
                if avg_correlation < 0.3:
                    st.info("Good Diversification")
                elif avg_correlation < 0.7:
                    st.warning("Moderate Diversification")
                else:
                    st.error("Poor Diversification")
            
            with col3:
                interest_rate_sensitivity = "Medium"
                long_term_etfs = ["TLT", "VGLT", "BLV", "SPTL"]
                short_term_etfs = ["SHY", "VCSH", "BSV", "SCHO", "VGSH", "SHV"]
                
                # Check if any long-term bonds are in the selected ETFs
                if any(etf in etf_names for etf in long_term_etfs):
                    interest_rate_sensitivity = "High"
                # Check if mostly short-term bonds
                elif all(etf in short_term_etfs for etf in etf_names):
                    interest_rate_sensitivity = "Low"
                
                st.metric("Interest Rate Sensitivity", interest_rate_sensitivity)
                
                if interest_rate_sensitivity == "Low":
                    st.info("Low sensitivity to interest rate changes")
                elif interest_rate_sensitivity == "Medium":
                    st.warning("Moderate sensitivity to interest rate changes")
                else:
                    st.error("High sensitivity to interest rate changes")
            
            # Portfolio Recommendation
            st.subheader("Portfolio Recommendation")
            
            if risk_score < 40:
                recommendation = "This conservative portfolio is well-suited for risk-averse investors. It offers stability with modest returns."
            elif risk_score < 70:
                recommendation = "This balanced portfolio offers a good compromise between risk and return, suitable for moderate investors."
            else:
                recommendation = "This aggressive portfolio offers higher potential returns but with increased volatility. Suitable for risk-tolerant investors."
            
            st.write(recommendation)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Try selecting different ETFs or check your API key.")

if __name__ == "__main__":
    run_portfolio_optimization()
