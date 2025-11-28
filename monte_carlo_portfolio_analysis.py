"""
Portfolio Monte Carlo Simulation Script
Run this to analyze your investment portfolio with 10,000 simulations

Usage: python monte_carlo_portfolio_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Asset class correlation assumptions
# These define how different asset classes correlate with each other
ASSET_CLASS_CORRELATIONS = {
    # Format: (class1, class2): correlation
    ('balanced', 'balanced'): 1.00,
    ('balanced', 'us_equity'): 0.85,
    ('balanced', 'us_equity_tech'): 0.75,
    ('balanced', 'international_equity'): 0.70,
    ('balanced', 'bonds'): 0.20,
    ('balanced', 'cash'): 0.00,

    ('us_equity', 'us_equity'): 1.00,
    ('us_equity', 'us_equity_tech'): 0.75,
    ('us_equity', 'international_equity'): 0.60,
    ('us_equity', 'bonds'): 0.15,
    ('us_equity', 'cash'): 0.00,

    ('us_equity_tech', 'us_equity_tech'): 1.00,
    ('us_equity_tech', 'international_equity'): 0.50,
    ('us_equity_tech', 'bonds'): 0.10,
    ('us_equity_tech', 'cash'): 0.00,

    ('international_equity', 'international_equity'): 1.00,
    ('international_equity', 'bonds'): 0.15,
    ('international_equity', 'cash'): 0.00,

    ('bonds', 'bonds'): 1.00,
    ('bonds', 'cash'): 0.10,

    ('cash', 'cash'): 1.00,
}

# Within same asset class, add slight variation for different holdings
SAME_CLASS_BASE_CORRELATION = 0.90  # High but not perfect correlation


def build_correlation_matrix(portfolio):
    """
    Dynamically build correlation matrix based on portfolio asset classes
    This keeps correlation logic separate from specific holdings
    """
    tickers = list(portfolio.keys())
    n = len(tickers)
    corr_matrix = np.eye(n)  # Start with identity matrix

    for i in range(n):
        for j in range(i+1, n):  # Only fill upper triangle
            class_i = portfolio[tickers[i]]['asset_class']
            class_j = portfolio[tickers[j]]['asset_class']

            if class_i == class_j:
                # Same asset class: high correlation with slight variation
                hash_val = abs(hash(tickers[i] + tickers[j]))
                variation = (hash_val % 5) / 100  # 0.00 to 0.04
                corr = min(0.95, SAME_CLASS_BASE_CORRELATION + variation)
            else:
                # Different asset classes: use lookup table
                key = tuple(sorted([class_i, class_j]))
                corr = ASSET_CLASS_CORRELATIONS.get(key, 0.50)

            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr  # Make symmetric

    # Ensure positive definite by checking eigenvalues
    min_eig = np.min(np.real(np.linalg.eigvals(corr_matrix)))
    if min_eig < 1e-10:
        # Add regularization to make positive definite
        corr_matrix += np.eye(n) * (abs(min_eig) + 0.01)

    return corr_matrix

# ============================================================================
# CONFIGURATION - PORTFOLIO LOADED FROM EXTERNAL FILE
# ============================================================================

try:
    from portfolio_config import portfolio
    print("✓ Loaded portfolio from portfolio_config.py")
except ImportError:
    print("⚠ portfolio_config.py not found. Using example portfolio.")
    print("  Copy portfolio_config.example.py to portfolio_config.py and update with your values.")

    # Example/fallback portfolio
    portfolio = {
        'SPY': {
            'value': 10000.00,
            'annual_return': 0.10,
            'volatility': 0.18,
            'beta': 1.0,
            'asset_class': 'us_equity'
        },
        'BND': {
            'value': 5000.00,
            'annual_return': 0.04,
            'volatility': 0.05,
            'beta': 0.10,
            'asset_class': 'bonds'
        },
        'VTI': {
            'value': 15000.00,
            'annual_return': 0.105,
            'volatility': 0.18,
            'beta': 1.0,
            'asset_class': 'us_equity'
        }
    }

# Simulation parameters
NUM_SIMULATIONS = 10000
TIME_HORIZON_YEARS = 10
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.04  # 4% risk-free rate (10-year Treasury)
ENABLE_REBALANCING = False  # Set True to rebalance quarterly
REBALANCE_FREQUENCY = 63  # Trading days (quarterly)

# ============================================================================
# ANALYSIS CODE
# ============================================================================

def calculate_portfolio_metrics(portfolio):
    """Calculate overall portfolio statistics"""
    total_value = sum(holding['value'] for holding in portfolio.values())
    
    # Calculate weights
    for ticker in portfolio:
        portfolio[ticker]['weight'] = portfolio[ticker]['value'] / total_value
    
    # Weighted metrics
    portfolio_return = sum(data['weight'] * data['annual_return'] 
                          for data in portfolio.values())
    
    # Simple volatility calculation (assumes independence - conservative estimate)
    portfolio_volatility = np.sqrt(
        sum((data['weight'] * data['volatility'])**2 
            for data in portfolio.values())
    )
    
    portfolio_beta = sum(data['weight'] * data['beta']
                        for data in portfolio.values())

    return {
        'total_value': total_value,
        'expected_return': portfolio_return,
        'volatility': portfolio_volatility,
        'beta': portfolio_beta,
        'sharpe_ratio': (portfolio_return - RISK_FREE_RATE) / portfolio_volatility
    }


def run_monte_carlo_simulation(portfolio, portfolio_metrics, num_sims, years):
    """Run Monte Carlo simulation with correlations"""
    initial_value = portfolio_metrics['total_value']
    num_days = years * TRADING_DAYS_PER_YEAR

    # Get tickers in order matching correlation matrix
    tickers = list(portfolio.keys())
    n_assets = len(tickers)

    # Extract returns and volatilities
    returns = np.array([portfolio[t]['annual_return'] for t in tickers])
    volatilities = np.array([portfolio[t]['volatility'] for t in tickers])
    weights = np.array([portfolio[t]['weight'] for t in tickers])

    # Build correlation matrix dynamically based on asset classes
    correlation_matrix = build_correlation_matrix(portfolio)

    # Cholesky decomposition for correlated returns
    cholesky = np.linalg.cholesky(correlation_matrix)

    final_values = []
    all_paths = []  # Store full paths for drawdown analysis

    print(f"Running {num_sims:,} simulations with correlated returns...")

    for sim in range(num_sims):
        if sim % 1000 == 0:
            print(f"  Completed {sim:,}/{num_sims:,} simulations...")

        # Track individual asset values for rebalancing
        asset_values = weights * initial_value
        path = [initial_value]

        for day in range(num_days):
            # Generate independent standard normal returns
            independent_returns = np.random.randn(n_assets)

            # Apply correlation via Cholesky decomposition
            correlated_returns = cholesky @ independent_returns

            # Convert to actual daily returns using GBM
            daily_returns = (returns / TRADING_DAYS_PER_YEAR) + \
                          (volatilities / np.sqrt(TRADING_DAYS_PER_YEAR)) * correlated_returns

            # Update asset values using geometric returns
            asset_values *= np.exp(daily_returns)

            # Rebalancing logic
            if ENABLE_REBALANCING and day > 0 and day % REBALANCE_FREQUENCY == 0:
                total_value = asset_values.sum()
                asset_values = weights * total_value

            path.append(asset_values.sum())

        final_values.append(asset_values.sum())
        all_paths.append(path)

    return np.array(final_values), np.array(all_paths)


def calculate_drawdown(path):
    """Calculate maximum drawdown for a single path"""
    peak = path[0]
    max_dd = 0
    for value in path:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def calculate_statistics(final_values, all_paths, initial_value, years):
    """Calculate key statistics from simulation results"""
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    results = np.percentile(final_values, percentiles)

    # Calculate drawdowns for all paths
    drawdowns = [calculate_drawdown(path) for path in all_paths]
    median_drawdown = np.median(drawdowns)
    max_drawdown = np.max(drawdowns)
    p95_drawdown = np.percentile(drawdowns, 95)

    stats = {
        'percentiles': {p: results[i] for i, p in enumerate(percentiles)},
        'expected_value': np.mean(final_values),
        'std_dev': np.std(final_values),
        'prob_loss': (final_values < initial_value).sum() / len(final_values) * 100,
        'prob_double': (final_values > 2 * initial_value).sum() / len(final_values) * 100,
        'prob_triple': (final_values > 3 * initial_value).sum() / len(final_values) * 100,
        'var_95': initial_value - np.percentile(final_values, 5),
        'cvar_95': initial_value - np.mean(final_values[final_values < np.percentile(final_values, 5)]),
        'median_drawdown': median_drawdown,
        'max_drawdown': max_drawdown,
        'p95_drawdown': p95_drawdown
    }

    return stats


def export_to_csv(portfolio, metrics, stats, final_values, years):
    """Export simulation results to CSV"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Portfolio summary
    summary_data = {
        'Metric': ['Total Value', 'Expected Return', 'Volatility', 'Beta', 'Sharpe Ratio'],
        'Value': [
            f"${metrics['total_value']:,.2f}",
            f"{metrics['expected_return']*100:.2f}%",
            f"{metrics['volatility']*100:.2f}%",
            f"{metrics['beta']:.2f}",
            f"{metrics['sharpe_ratio']:.3f}"
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(f'portfolio_summary_{timestamp}.csv', index=False)

    # Holdings breakdown
    holdings_data = []
    for ticker, data in portfolio.items():
        holdings_data.append({
            'Ticker': ticker,
            'Value': data['value'],
            'Weight': data['weight'] * 100,
            'Annual_Return': data['annual_return'] * 100,
            'Volatility': data['volatility'] * 100,
            'Beta': data['beta'],
            'Asset_Class': data['asset_class']
        })
    df_holdings = pd.DataFrame(holdings_data)
    df_holdings.to_csv(f'portfolio_holdings_{timestamp}.csv', index=False)

    # Simulation results
    simulation_data = {
        'Percentile': list(stats['percentiles'].keys()),
        'Value': [f"${v:,.2f}" for v in stats['percentiles'].values()],
        'Gain': [f"${v - metrics['total_value']:,.2f}" for v in stats['percentiles'].values()],
        'Gain_Pct': [f"{(v/metrics['total_value'] - 1)*100:.2f}%" for v in stats['percentiles'].values()],
        'CAGR': [f"{((v/metrics['total_value'])**(1/years) - 1)*100:.2f}%" for v in stats['percentiles'].values()]
    }
    df_simulation = pd.DataFrame(simulation_data)
    df_simulation.to_csv(f'simulation_results_{timestamp}.csv', index=False)

    # Risk metrics
    risk_data = {
        'Metric': [
            'Expected Value', 'Std Deviation', 'Prob Loss', 'Prob Double', 'Prob Triple',
            'VaR 95%', 'CVaR 95%', 'Median Drawdown', '95th% Drawdown', 'Max Drawdown'
        ],
        'Value': [
            f"${stats['expected_value']:,.2f}",
            f"${stats['std_dev']:,.2f}",
            f"{stats['prob_loss']:.2f}%",
            f"{stats['prob_double']:.2f}%",
            f"{stats['prob_triple']:.2f}%",
            f"${stats['var_95']:,.2f}",
            f"${stats['cvar_95']:,.2f}",
            f"{stats['median_drawdown']*100:.2f}%",
            f"{stats['p95_drawdown']*100:.2f}%",
            f"{stats['max_drawdown']*100:.2f}%"
        ]
    }
    df_risk = pd.DataFrame(risk_data)
    df_risk.to_csv(f'risk_metrics_{timestamp}.csv', index=False)

    print(f"\n✓ CSV files exported:")
    print(f"  - portfolio_summary_{timestamp}.csv")
    print(f"  - portfolio_holdings_{timestamp}.csv")
    print(f"  - simulation_results_{timestamp}.csv")
    print(f"  - risk_metrics_{timestamp}.csv")


def print_results(portfolio, metrics, stats, years):
    """Print detailed results"""
    print("\n" + "="*70)
    print("PORTFOLIO ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\nTotal Portfolio Value: ${metrics['total_value']:,.2f}")
    print("\nCurrent Allocation:")
    print("-"*70)
    for ticker, data in portfolio.items():
        print(f"{ticker:8s} ${data['value']:>10,.2f}  ({data['weight']*100:>5.1f}%)  "
              f"Return: {data['annual_return']*100:>5.1f}%  Vol: {data['volatility']*100:>5.1f}%")
    
    print("\n" + "="*70)
    print("PORTFOLIO METRICS")
    print("="*70)
    print(f"Expected Annual Return: {metrics['expected_return']*100:.2f}%")
    print(f"Portfolio Volatility: {metrics['volatility']*100:.2f}%")
    print(f"Portfolio Beta: {metrics['beta']:.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    
    print("\n" + "="*70)
    print(f"MONTE CARLO RESULTS ({years} years)")
    print("="*70)
    
    print(f"\nProjected Portfolio Values:")
    print("-"*70)
    for percentile, value in stats['percentiles'].items():
        gain = value - metrics['total_value']
        pct_gain = (value / metrics['total_value'] - 1) * 100
        annualized = ((value / metrics['total_value']) ** (1/years) - 1) * 100
        print(f"{percentile:2d}th Percentile: ${value:>12,.2f}  "
              f"(+${gain:>10,.2f}, {pct_gain:>6.1f}%, Ann: {annualized:>5.2f}%)")
    
    print(f"\nExpected Value: ${stats['expected_value']:,.2f}")
    print(f"Standard Deviation: ${stats['std_dev']:,.2f}")
    
    print(f"\nProbability Analysis:")
    print(f"  Loss after {years} years: {stats['prob_loss']:.2f}%")
    print(f"  Doubling: {stats['prob_double']:.2f}%")
    print(f"  Tripling: {stats['prob_triple']:.2f}%")
    
    print(f"\nRisk Metrics:")
    print(f"  Value at Risk (95%): ${stats['var_95']:,.2f}")
    print(f"  Conditional VaR (95%): ${stats['cvar_95']:,.2f}")
    print(f"  Median Max Drawdown: {stats['median_drawdown']*100:.2f}%")
    print(f"  Worst Drawdown (95th percentile): {stats['p95_drawdown']*100:.2f}%")
    print(f"  Absolute Worst Drawdown: {stats['max_drawdown']*100:.2f}%")


def create_visualizations(portfolio, final_values, metrics, years):
    """Create visualization charts"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribution histogram
    axes[0, 0].hist(final_values, bins=100, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(metrics['total_value'], color='red', 
                       linestyle='--', linewidth=2, label='Initial Value')
    axes[0, 0].axvline(np.mean(final_values), color='green', 
                       linestyle='--', linewidth=2, label='Expected Value')
    axes[0, 0].set_xlabel('Portfolio Value ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Distribution of Portfolio Values After {years} Years')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Allocation pie chart
    colors = plt.cm.Set3(range(len(portfolio)))
    axes[0, 1].pie(
        [data['value'] for data in portfolio.values()],
        labels=[f"{ticker}\n${data['value']:,.0f}\n({data['weight']*100:.1f}%)" 
                for ticker, data in portfolio.items()],
        colors=colors,
        startangle=90
    )
    axes[0, 1].set_title('Current Portfolio Allocation')
    
    # 3. Sample simulation paths
    num_paths = 50
    time_points = np.linspace(0, years, 100)
    
    for _ in range(num_paths):
        path = [metrics['total_value']]
        value = metrics['total_value']
        
        for i in range(1, len(time_points)):
            days = int((time_points[i] - time_points[i-1]) * TRADING_DAYS_PER_YEAR)
            for _ in range(days):
                daily_return = np.random.normal(
                    metrics['expected_return'] / TRADING_DAYS_PER_YEAR,
                    metrics['volatility'] / np.sqrt(TRADING_DAYS_PER_YEAR)
                )
                value *= (1 + daily_return)
            path.append(value)
        
        axes[1, 0].plot(time_points, path, alpha=0.3, linewidth=0.5)
    
    axes[1, 0].axhline(metrics['total_value'], color='red', 
                       linestyle='--', linewidth=2, label='Initial Value')
    axes[1, 0].set_xlabel('Years')
    axes[1, 0].set_ylabel('Portfolio Value ($)')
    axes[1, 0].set_title(f'Sample Simulation Paths (n={num_paths})')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Risk-Return scatter
    returns = [data['annual_return'] for data in portfolio.values()]
    volatilities = [data['volatility'] for data in portfolio.values()]
    sizes = [data['value']/100 for data in portfolio.values()]
    tickers = list(portfolio.keys())
    
    axes[1, 1].scatter(volatilities, returns, s=sizes, alpha=0.6, 
                      c=range(len(tickers)), cmap='viridis')
    
    for i, ticker in enumerate(tickers):
        axes[1, 1].annotate(ticker, (volatilities[i], returns[i]),
                           xytext=(5, 5), textcoords='offset points')
    
    # Add portfolio point
    axes[1, 1].scatter([metrics['volatility']], [metrics['expected_return']],
                      s=metrics['total_value']/50, marker='*', c='red',
                      label='Portfolio', edgecolors='black', linewidths=2)
    
    axes[1, 1].set_xlabel('Volatility (Risk)')
    axes[1, 1].set_ylabel('Expected Return')
    axes[1, 1].set_title('Risk-Return Profile (bubble size = value)')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('portfolio_monte_carlo_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved as 'portfolio_monte_carlo_analysis.png'")
    
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main analysis function"""
    print("\n" + "="*70)
    print("PORTFOLIO MONTE CARLO SIMULATION")
    print("="*70)
    
    # Calculate portfolio metrics
    metrics = calculate_portfolio_metrics(portfolio)
    
    # Run Monte Carlo simulation
    final_values, all_paths = run_monte_carlo_simulation(
        portfolio,
        metrics,
        NUM_SIMULATIONS,
        TIME_HORIZON_YEARS
    )

    # Calculate statistics
    stats = calculate_statistics(
        final_values,
        all_paths,
        metrics['total_value'],
        TIME_HORIZON_YEARS
    )

    # Print results
    print_results(portfolio, metrics, stats, TIME_HORIZON_YEARS)

    # Export to CSV
    export_to_csv(portfolio, metrics, stats, final_values, TIME_HORIZON_YEARS)

    # Create visualizations
    create_visualizations(portfolio, final_values, metrics, TIME_HORIZON_YEARS)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the visualization: portfolio_monte_carlo_analysis.png")
    print("2. Consider the rebalancing recommendations")
    print("3. Adjust portfolio values in the script and re-run as needed")
    print("\n")


if __name__ == "__main__":
    main()