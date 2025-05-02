import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

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
    'IGSB': 'iShares Short-Term Corporate Bond ETF'
}

# Simplified to 20 ETFs to make the list more manageable

# Function to generate mock data for ETFs
def generate_mock_data(tickers, days=252):
    """
    Generate mock price data for the selected ETFs
    """
    np.random.seed(42)  # For reproducibility
    
    # Start date (approximately 1 year ago)
    start_date = datetime.now() - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Create DataFrame with dates as index
    df = pd.DataFrame(index=dates)
    
    # Average expected returns and volatilities for different types of bond ETFs
    etf_characteristics = {
        # Ticker: (annual return, annual volatility)
        'AGG': (0.03, 0.05),  # Core aggregate bonds
        'BND': (0.03, 0.05),  # Total bond market
        'VCIT': (0.035, 0.06),  # Intermediate corporate
        'LQD': (0.035, 0.06),  # Investment grade corporate
        'VCSH': (0.025, 0.03),  # Short-term corporate
        'BSV': (0.02, 0.02),  # Short-term bonds
        'MBB': (0.025, 0.04),  # Mortgage-backed
        'GOVT': (0.022, 0.04),  # US Treasury
        'VTIP': (0.02, 0.03),  # Short-term TIPS
        'TIP': (0.025, 0.045),  # TIPS
        'IGIB': (0.033, 0.055),  # Intermediate corporate
        'MUB': (0.028, 0.04),  # Municipal bonds
        'VGSH': (0.018, 0.015),  # Short-term Treasury
        'VGIT': (0.022, 0.035),  # Intermediate Treasury
        'HYG': (0.05, 0.08),  # High yield corporate
        'VMBS': (0.024, 0.035),  # Mortgage-backed
        'IEF': (0.023, 0.055),  # 7-10 Year Treasury
        'SHV': (0.015, 0.01),  # Short Treasury
        'VTEB': (0.027, 0.035),  # Tax-exempt
        'IGSB': (0.025, 0.025)  # Short-term corporate
    }
    
    # Generate price data for each ticker
    for ticker in tickers:
        # Get characteristics or use default values
        annual_return, annual_volatility = etf_characteristics.get(ticker, (0.03, 0.05))
        
        # Convert annual to daily parameters
        daily_return = annual_return / 252
        daily_volatility = annual_volatility / np.sqrt(252)
        
        # Generate random returns
        returns = np.random.normal(daily_return, daily_volatility, days)
        
        # Create price series starting at 100
        prices = 100 * (1 + returns).cumprod()
        
        # Add to DataFrame
        df[ticker] = prices
    
    # Add correlation between ETFs (bonds tend to be correlated)
    # Create a correlation matrix
    corr_matrix = np.zeros((len(tickers), len(tickers)))
    
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            # High correlation between similar types of bonds
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                # Base correlation is 0.7, modified by difference in volatility
                ticker_i = tickers[i]
                ticker_j = tickers[j]
                vol_i = etf_characteristics.get(ticker_i, (0.03, 0.05))[1]
                vol_j = etf_characteristics.get(ticker_j, (0.03, 0.05))[1]
                
                # More similar volatilities = higher correlation
                sim_factor = 1 - abs(vol_i - vol_j) / max(vol_i, vol_j)
                corr_matrix[i, j] = 0.7 * sim_factor
    
    # Generate correlated returns
    means = np.array([etf_characteristics.get(ticker, (0.03, 0.05))[0] / 252 for ticker in tickers])
    stds = np.array([etf_characteristics.get(ticker, (0.03, 0.05))[1] / np.sqrt(252) for ticker in tickers])
    
    # Generate correlated normal random variables
    L = np.linalg.cholesky(corr_matrix)
    uncorrelated = np.random.normal(0, 1, size=(days, len(tickers)))
    correlated = uncorrelated @ L.T
    
    # Scale to appropriate mean and standard deviation
    for i in range(len(tickers)):
        correlated[:, i] = means[i] + correlated[:, i] * stds[i]
    
    # Convert to prices
    df = pd.DataFrame(100 * (1 + correlated).cumprod(), columns=tickers, index=dates)
    
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
    
    *Note: This app uses simulated data based on historical characteristics of these ETFs.*
    """)
    
    # Sidebar for ETF selection
    st.sidebar.header("ETF Selection")
    
    # Default ETFs
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
    days = st.sidebar.selectbox(
        "Historical Period (Days):",
        [252, 504, 756],  # 1 year, 2 years, 3 years
        index=0,
        format_func=lambda x: f"{x//252} year{'s' if x//252 > 1 else ''} ({x} trading days)"
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
            # Generate mock data
            prices = generate_mock_data(selected_etfs, days)
            
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
                title=f"Historical Price Performance (Simulated)",
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
            
            # Disclaimer
            st.caption("**Disclaimer**: This application uses simulated data based on historical ETF characteristics. The results are for educational purposes only and should not be considered investment advice.")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    run_portfolio_optimization()
