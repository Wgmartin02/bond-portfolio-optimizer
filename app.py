import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# List of top 50 bond ETFs by AUM
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


def get_historical_data(tickers, period='2y'):
    """
    Fetch historical price data for the given tickers
    """
    try:
        data = yf.download(tickers, period=period, interval='1d')['Adj Close']
        
        # If we only have one ticker, the result is a Series, not a DataFrame
        # Convert Series to DataFrame for consistency
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)
            data.columns = [tickers]
            
        # Check if we have actual data
        if data.empty:
            return None
            
        # Check for columns with all NaN values
        valid_columns = data.columns[data.notna().any()]
        if len(valid_columns) == 0:
            return None
            
        return data[valid_columns]
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None


def calculate_returns(prices):
    """
    Calculate daily returns and annualized metrics from price data
    """
    if prices is None or prices.empty:
        return None, None, None
        
    try:
        daily_returns = prices.pct_change().dropna()
        
        # Handle case with insufficient data
        if len(daily_returns) < 20:  # Need enough data points for meaningful statistics
            return None, None, None
            
        annual_returns = (1 + daily_returns.mean()) ** 252 - 1
        annual_volatility = daily_returns.std() * np.sqrt(252)
        return daily_returns, annual_returns, annual_volatility
    except Exception as e:
        st.error(f"Error calculating returns: {str(e)}")
        return None, None, None


def calculate_sharpe_ratio(returns, volatility, risk_free_rate=0.03):
    """
    Calculate the Sharpe ratio for a portfolio
    """
    if returns is None or volatility is None or volatility == 0:
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
    
    try:
        for _ in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            
            # Calculate annualized portfolio return and volatility
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
    except Exception as e:
        st.error(f"Error in Monte Carlo simulation: {str(e)}")
        return pd.DataFrame()


def optimize_portfolio(returns, risk_free_rate=0.03):
    """
    Find the optimal portfolio weights to maximize the Sharpe ratio
    """
    if returns is None or returns.empty or len(returns.columns) < 2:
        return None
        
    num_assets = len(returns.columns)
    args = (returns.mean() * 252, returns.cov() * 252, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = np.array([1/num_assets] * num_assets)
    
    try:
        optimal_weights = minimize(
            negative_sharpe_ratio, 
            initial_guess, 
            args=args, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
        
        if optimal_weights['success']:
            return optimal_weights['x']
        else:
            st.warning("Optimization did not converge. Using equal weights instead.")
            return np.array([1/num_assets] * num_assets)
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        return np.array([1/num_assets] * num_assets)


def create_efficient_frontier_chart(mc_results, optimal_portfolio, etf_names):
    """
    Create a plotly chart showing the efficient frontier and optimal portfolio
    """
    if mc_results.empty:
        return None
        
    try:
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
    except Exception as e:
        st.error(f"Error creating efficient frontier chart: {str(e)}")
        return None


def create_portfolio_composition_chart(weights, etf_names):
    """
    Create a plotly pie chart showing portfolio composition
    """
    if weights is None or len(weights) == 0:
        return None
        
    try:
        # Round weights to ensure they sum to 100%
        weights = np.round(weights * 100, 2)
        
        fig = px.pie(
            values=weights,
            names=etf_names,
            title='Optimal Portfolio Composition'
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        
        return fig
    except Exception as e:
        st.error(f"Error creating composition chart: {str(e)}")
        return None


def create_correlation_heatmap(returns, etf_names):
    """
    Create a plotly heatmap showing correlation between assets
    """
    if returns is None or returns.empty:
        return None
        
    try:
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
    except Exception as e:
        st.error(f"Error creating correlation heatmap: {str(e)}")
        return None


def create_risk_metrics_table(optimal_portfolio, etf_names):
    """
    Create a risk metrics table for the optimal portfolio
    """
    if optimal_portfolio is None:
        return None, None
        
    try:
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
    except Exception as e:
        st.error(f"Error creating metrics table: {str(e)}")
        return None, None


def run_ai_portfolio_optimization():
    st.set_page_config(layout="wide", page_title="AI-Powered Bond Portfolio Optimizer")
    
    st.title("AI-Powered Bond Portfolio Optimizer")
    st.write("""
    This application helps optimize a bond ETF portfolio. Select 2-4 bond ETFs from the list,
    and the AI-powered system will find the optimal allocation to maximize risk-adjusted returns.
    """)
    
    # Sidebar for ETF selection
    st.sidebar.header("ETF Selection")
    
    # Default ETFs that are more likely to have good data
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
        ["1y", "2y", "3y", "5y"],
        index=1
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
    
    try:
        with st.spinner("Fetching ETF data and analyzing portfolio..."):
            # Get historical data
            prices = get_historical_data(selected_etfs, period=time_period)
            
            # Check if we have valid data
            if prices is None or prices.empty:
                st.error("Could not retrieve data for the selected ETFs. Please try different ETFs.")
                st.stop()
                
            # Check if we have enough columns (ETFs)
            if len(prices.columns) < 2:
                st.error(f"Only retrieved data for {len(prices.columns)} ETF(s). Need at least 2 ETFs with data.")
                st.stop()
                
            # Show which ETFs have data
            valid_etfs = list(prices.columns)
            if len(valid_etfs) < len(selected_etfs):
                st.warning(f"Only found data for these ETFs: {', '.join(valid_etfs)}")
                
            # Calculate returns and metrics
            daily_returns, annual_returns, annual_volatility = calculate_returns(prices)
            
            if daily_returns is None or annual_returns is None:
                st.error("Could not calculate returns from the data. Try different ETFs or a longer time period.")
                st.stop()
                
            # Run Monte Carlo simulation
            mc_results = monte_carlo_simulation(daily_returns)
            
            if mc_results.empty:
                st.error("Monte Carlo simulation failed. Try different ETFs.")
                st.stop()
                
            # Find optimal portfolio
            optimal_weights = optimize_portfolio(daily_returns, risk_free_rate)
            
            if optimal_weights is None:
                st.error("Portfolio optimization failed. Try different ETFs.")
                st.stop()
                
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
            etf_names = valid_etfs
            
            # ---- Display Results ----
            col1, col2 = st.columns([3, 1])
            
            # Efficient Frontier Chart
            with col1:
                efficient_frontier = create_efficient_frontier_chart(mc_results, optimal_portfolio, etf_names)
                if efficient_frontier:
                    st.plotly_chart(efficient_frontier, use_container_width=True)
            
            # Portfolio Composition
            with col2:
                composition_chart = create_portfolio_composition_chart(optimal_portfolio['weights'], etf_names)
                if composition_chart:
                    st.plotly_chart(composition_chart, use_container_width=True)
            
            # Risk Metrics Table
            st.subheader("Portfolio Risk Metrics")
            metrics_df, summary_df = create_risk_metrics_table(optimal_portfolio, etf_names)
            
            if metrics_df is not None and summary_df is not None:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.table(metrics_df)
                with col2:
                    st.table(summary_df)
            
            # Correlation Heatmap
            st.subheader("ETF Correlation Analysis")
            correlation_chart = create_correlation_heatmap(daily_returns, etf_names)
            if correlation_chart:
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
            
            # Simple risk score based on volatility and correlation
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
                
                # Check if any long-term bonds are in the selected ETFs with significant weight
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
            
            # Bond Market Outlook
            st.subheader("Market Outlook and AI Recommendation")
            
            if risk_score < 40:
                recommendation = "This conservative portfolio is well-suited for risk-averse investors. It offers stability with modest returns."
            elif risk_score < 70:
                recommendation = "This balanced portfolio offers a good compromise between risk and return, suitable for moderate investors."
            else:
                recommendation = "This aggressive portfolio offers higher potential returns but with increased volatility. Suitable for risk-tolerant investors."
            
            st.write(recommendation)
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.write("Try selecting different ETFs or a different time period.")


if __name__ == "__main__":
    run_ai_portfolio_optimization()
