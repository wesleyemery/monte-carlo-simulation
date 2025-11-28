"""
Interactive Monte Carlo Portfolio Dashboard
Run with: streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import portfolio configuration
try:
    from portfolio_config import portfolio as default_portfolio
except ImportError:
    default_portfolio = {
        'SPY': {'value': 10000, 'annual_return': 0.10, 'volatility': 0.18, 'beta': 1.0, 'asset_class': 'us_equity'},
        'BND': {'value': 5000, 'annual_return': 0.04, 'volatility': 0.05, 'beta': 0.10, 'asset_class': 'bonds'},
        'VTI': {'value': 15000, 'annual_return': 0.105, 'volatility': 0.18, 'beta': 1.0, 'asset_class': 'us_equity'}
    }

# Import correlation building function
from monte_carlo_portfolio_analysis import (
    build_correlation_matrix,
    calculate_portfolio_metrics,
    calculate_drawdown,
    ASSET_CLASS_CORRELATIONS,
    SAME_CLASS_BASE_CORRELATION
)

# Page config
st.set_page_config(
    page_title="Portfolio Monte Carlo Simulator",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Portfolio Monte Carlo Simulation Dashboard")
st.markdown("Interactive visualization of portfolio projections with customizable parameters")

# Sidebar controls
st.sidebar.header("⚙️ Simulation Parameters")

# Portfolio total value
total_value = sum(h['value'] for h in default_portfolio.values())
st.sidebar.metric("Current Portfolio Value", f"${total_value:,.2f}")

# Simulation parameters
num_sims = st.sidebar.slider("Number of Simulations", 1000, 50000, 10000, 1000)
years = st.sidebar.slider("Time Horizon (Years)", 1, 40, 10)
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.0, 0.1) / 100

st.sidebar.subheader("Monthly Contributions")
enable_contributions = st.sidebar.checkbox("Enable Monthly Contributions", False)
monthly_contribution = st.sidebar.number_input(
    "Monthly Amount ($)",
    min_value=0,
    max_value=50000,
    value=1000 if enable_contributions else 0,
    step=100
) if enable_contributions else 0

contribution_growth = st.sidebar.slider(
    "Annual Contribution Growth (%)",
    0.0, 10.0, 3.0, 0.5
) / 100 if enable_contributions else 0

st.sidebar.subheader("Portfolio Management")
enable_rebalancing = st.sidebar.checkbox("Enable Quarterly Rebalancing", False)

# Run simulation button
run_simulation = st.sidebar.button("🚀 Run Simulation", type="primary")

# Constants
TRADING_DAYS_PER_YEAR = 252

def run_monte_carlo(portfolio, num_sims, years, monthly_contrib, contrib_growth, rebalance):
    """Run Monte Carlo simulation"""
    metrics = calculate_portfolio_metrics(portfolio)
    metrics['sharpe_ratio'] = (metrics['expected_return'] - risk_free_rate) / metrics['volatility']

    initial_value = metrics['total_value']
    num_days = years * TRADING_DAYS_PER_YEAR

    tickers = list(portfolio.keys())
    n_assets = len(tickers)

    returns = np.array([portfolio[t]['annual_return'] for t in tickers])
    volatilities = np.array([portfolio[t]['volatility'] for t in tickers])
    weights = np.array([portfolio[t]['weight'] for t in tickers])

    correlation_matrix = build_correlation_matrix(portfolio)
    cholesky = np.linalg.cholesky(correlation_matrix)

    final_values = []
    all_paths = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for sim in range(num_sims):
        if sim % max(1, num_sims // 20) == 0:
            progress_bar.progress(sim / num_sims)
            status_text.text(f"Running simulation {sim:,}/{num_sims:,}...")

        asset_values = weights * initial_value
        path = [initial_value]
        current_contribution = monthly_contrib

        for day in range(num_days):
            independent_returns = np.random.randn(n_assets)
            correlated_returns = cholesky @ independent_returns
            daily_returns = (returns / TRADING_DAYS_PER_YEAR) + \
                          (volatilities / np.sqrt(TRADING_DAYS_PER_YEAR)) * correlated_returns

            asset_values *= np.exp(daily_returns)

            # Contributions
            if monthly_contrib > 0 and day > 0 and day % 21 == 0:
                asset_values += weights * current_contribution

            if monthly_contrib > 0 and day > 0 and day % TRADING_DAYS_PER_YEAR == 0:
                current_contribution *= (1 + contrib_growth)

            # Rebalancing
            if rebalance and day > 0 and day % 63 == 0:
                total_val = asset_values.sum()
                asset_values = weights * total_val

            path.append(asset_values.sum())

        final_values.append(asset_values.sum())
        all_paths.append(path)

    progress_bar.progress(1.0)
    status_text.text("✓ Simulation complete!")

    return np.array(final_values), np.array(all_paths), metrics

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

# Run simulation when button clicked
if run_simulation:
    with st.spinner('Running Monte Carlo simulation...'):
        final_values, all_paths, metrics = run_monte_carlo(
            default_portfolio,
            num_sims,
            years,
            monthly_contribution,
            contribution_growth,
            enable_rebalancing
        )

        # Calculate statistics
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        percentile_values = np.percentile(final_values, percentiles)

        drawdowns = [calculate_drawdown(path) for path in all_paths]

        st.session_state.results = {
            'final_values': final_values,
            'all_paths': all_paths,
            'metrics': metrics,
            'percentiles': dict(zip(percentiles, percentile_values)),
            'drawdowns': drawdowns,
            'years': years,
            'monthly_contribution': monthly_contribution
        }

# Display results
if st.session_state.results:
    res = st.session_state.results
    metrics = res['metrics']
    final_values = res['final_values']

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Expected Return",
            f"{metrics['expected_return']*100:.2f}%",
            help="Weighted average annual return"
        )

    with col2:
        st.metric(
            "Portfolio Volatility",
            f"{metrics['volatility']*100:.2f}%",
            help="Expected annual standard deviation"
        )

    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.3f}",
            help="Risk-adjusted return (higher is better)"
        )

    with col4:
        median_final = res['percentiles'][50]
        initial = metrics['total_value']
        total_contrib = monthly_contribution * 12 * res['years'] if monthly_contribution > 0 else 0
        gain_pct = ((median_final - initial - total_contrib) / (initial + total_contrib)) * 100
        st.metric(
            "Median Gain",
            f"{gain_pct:.1f}%",
            help=f"50th percentile outcome over {res['years']} years"
        )

    # Main visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution", "📈 Projections", "⚠️ Risk Analysis", "💼 Portfolio"])

    with tab1:
        st.subheader(f"Distribution of Final Values ({res['years']}-Year Horizon)")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(final_values, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(metrics['total_value'], color='red', linestyle='--', linewidth=2, label='Initial Value')
        ax.axvline(np.mean(final_values), color='green', linestyle='--', linewidth=2, label='Expected Value')
        ax.axvline(res['percentiles'][50], color='orange', linestyle='--', linewidth=2, label='Median')
        ax.set_xlabel('Portfolio Value ($)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        # Percentile table
        st.subheader("Percentile Outcomes")
        percentile_df = pd.DataFrame([
            {
                'Percentile': f"{p}th",
                'Value': f"${v:,.0f}",
                'Gain': f"${v - metrics['total_value']:,.0f}",
                'CAGR': f"{((v / metrics['total_value']) ** (1/res['years']) - 1) * 100:.2f}%"
            }
            for p, v in res['percentiles'].items()
        ])
        st.dataframe(percentile_df, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Sample Simulation Paths")

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot sample paths
        sample_indices = np.random.choice(len(res['all_paths']), min(50, len(res['all_paths'])), replace=False)
        time_points = np.linspace(0, res['years'], len(res['all_paths'][0]))

        for idx in sample_indices:
            ax.plot(time_points, res['all_paths'][idx], alpha=0.2, linewidth=0.5, color='steelblue')

        # Plot percentile bands
        percentile_paths = np.percentile(res['all_paths'], [10, 50, 90], axis=0)
        ax.plot(time_points, percentile_paths[1], color='orange', linewidth=3, label='Median (50th)')
        ax.fill_between(time_points, percentile_paths[0], percentile_paths[2],
                        alpha=0.3, color='green', label='10th-90th Percentile Range')

        ax.axhline(metrics['total_value'], color='red', linestyle='--', linewidth=2, label='Initial Value')
        ax.set_xlabel('Years', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with tab3:
        st.subheader("Risk Metrics")

        col1, col2 = st.columns(2)

        with col1:
            prob_loss = (final_values < metrics['total_value']).sum() / len(final_values) * 100
            prob_double = (final_values > 2 * metrics['total_value']).sum() / len(final_values) * 100
            prob_triple = (final_values > 3 * metrics['total_value']).sum() / len(final_values) * 100

            st.metric("Probability of Loss", f"{prob_loss:.2f}%")
            st.metric("Probability of Doubling", f"{prob_double:.2f}%")
            st.metric("Probability of Tripling", f"{prob_triple:.2f}%")

        with col2:
            median_dd = np.median(res['drawdowns']) * 100
            p95_dd = np.percentile(res['drawdowns'], 95) * 100
            max_dd = np.max(res['drawdowns']) * 100

            st.metric("Median Max Drawdown", f"{median_dd:.2f}%")
            st.metric("95th Percentile Drawdown", f"{p95_dd:.2f}%")
            st.metric("Worst Drawdown", f"{max_dd:.2f}%")

        # Drawdown distribution
        st.subheader("Drawdown Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(np.array(res['drawdowns']) * 100, bins=50, edgecolor='black', alpha=0.7, color='crimson')
        ax.set_xlabel('Maximum Drawdown (%)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with tab4:
        st.subheader("Current Portfolio Allocation")

        # Holdings table
        holdings_df = pd.DataFrame([
            {
                'Ticker': ticker,
                'Value': f"${data['value']:,.2f}",
                'Weight': f"{data['weight']*100:.2f}%",
                'Return': f"{data['annual_return']*100:.1f}%",
                'Volatility': f"{data['volatility']*100:.1f}%",
                'Asset Class': data['asset_class']
            }
            for ticker, data in default_portfolio.items()
        ])
        st.dataframe(holdings_df, use_container_width=True, hide_index=True)

        # Pie chart
        fig, ax = plt.subplots(figsize=(10, 6))
        values = [data['value'] for ticker, data in default_portfolio.items()]
        labels = [f"{ticker}\n${data['value']:,.0f}\n({data['weight']*100:.1f}%)"
                 for ticker, data in default_portfolio.items()]
        ax.pie(values, labels=labels, autopct='', startangle=90)
        ax.set_title('Portfolio Allocation')
        st.pyplot(fig)

else:
    st.info("👈 Configure parameters in the sidebar and click 'Run Simulation' to begin")

    # Show portfolio preview
    st.subheader("Current Portfolio")
    preview_df = pd.DataFrame([
        {
            'Ticker': ticker,
            'Value': f"${data['value']:,.2f}",
            'Return': f"{data['annual_return']*100:.1f}%",
            'Volatility': f"{data['volatility']*100:.1f}%"
        }
        for ticker, data in default_portfolio.items()
    ])
    st.dataframe(preview_df, use_container_width=True, hide_index=True)
