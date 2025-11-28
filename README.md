# Portfolio Monte Carlo Simulation

Python-based Monte Carlo simulation for investment portfolio analysis and 10-year projections.

## Requirements

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy pandas matplotlib seaborn streamlit
```

## Quick Start

### Command Line (Traditional)

```bash
python monte_carlo_portfolio_analysis.py
```

The script will run 10,000 simulations and generate:
- Console output with statistics
- `portfolio_monte_carlo_analysis.png` with 4-panel visualization
- CSV files with detailed results

### Interactive Dashboard

```bash
streamlit run dashboard.py
```

Opens an interactive web dashboard where you can:
- Adjust simulation parameters with sliders
- Enable/disable monthly contributions in real-time
- Visualize results instantly with interactive charts
- Compare different scenarios side-by-side

## Portfolio Configuration

**Setup (keeps your data private):**

1. Copy the example config:
   ```bash
   cp portfolio_config.example.py portfolio_config.py
   ```

2. Edit `portfolio_config.py` with your actual holdings:
   ```python
   portfolio = {
       'TICKER': {
           'value': 10000.00,       # Current dollar value
           'annual_return': 0.10,   # Expected annual return (10%)
           'volatility': 0.18,      # Annual volatility (18%)
           'beta': 1.0,             # Market beta
           'asset_class': 'us_equity'  # Classification
       }
   }
   ```

3. `portfolio_config.py` is already in `.gitignore` - your data stays private!

**Asset classes:** `balanced`, `us_equity`, `us_equity_tech`, `international_equity`, `bonds`, `cash`

## Output

### Console
- Portfolio composition and metrics (expected return, volatility, beta, Sharpe ratio)
- Percentile projections (5th, 10th, 25th, 50th, 75th, 90th, 95th)
- Probability analysis (loss, doubling, tripling)
- Risk metrics (VaR, CVaR, maximum drawdown)

### CSV Files (timestamped)
- `portfolio_summary_YYYYMMDD_HHMMSS.csv` - Overall metrics
- `portfolio_holdings_YYYYMMDD_HHMMSS.csv` - Individual holdings breakdown
- `simulation_results_YYYYMMDD_HHMMSS.csv` - Percentile outcomes
- `risk_metrics_YYYYMMDD_HHMMSS.csv` - Comprehensive risk analysis

### Visualization
Four-panel chart (`portfolio_monte_carlo_analysis.png`):
1. Distribution of final values (histogram)
2. Current allocation (pie chart)
3. Sample simulation paths (50 trajectories)
4. Risk-return scatter plot (all holdings + portfolio)

## Methodology

Uses **Geometric Brownian Motion** with **correlated returns** to model realistic market behavior:

```
S(t+1) = S(t) × exp(daily_return)
daily_return = (μ/252) + (σ/√252) × Z_correlated
```

Where:
- μ = expected annual return
- σ = annual volatility
- Z_correlated = correlated standard normal returns (via Cholesky decomposition)
- 252 = trading days per year

**Key improvements:**
- **Dynamic correlation matrix** - automatically generated based on asset classes in your portfolio
- **Geometric returns** (exponential) instead of arithmetic for mathematical accuracy
- **Optional quarterly rebalancing** to maintain target weights
- **Drawdown analysis** tracks peak-to-trough losses during simulations

### How Correlations Work

The simulation builds a correlation matrix based on asset class relationships:
- **Same asset class**: 0.90+ correlation (e.g., two US equity funds move together)
- **Related classes**: 0.60-0.85 (e.g., US equity ↔ international equity)
- **Uncorrelated**: 0.00-0.15 (e.g., bonds ↔ tech stocks, cash ↔ everything)

No ticker-specific data is stored—only general market behavior patterns.

## Key Metrics

- **Percentiles:** Show distribution of outcomes (50th = median)
- **CAGR:** Compound Annual Growth Rate over the period
- **VaR (Value at Risk):** Maximum expected loss at 95% confidence
- **CVaR (Conditional VaR):** Expected loss in worst 5% of scenarios
- **Sharpe Ratio:** Excess return per unit of risk (uses 4% risk-free rate)
- **Maximum Drawdown:** Largest peak-to-trough decline during simulation period

## Limitations

- Correlation estimates are approximations based on asset classes
- No transaction costs or taxes modeled
- Contributions are optional but don't account for contribution limits (401k/IRA caps)
- Parameters (returns, volatility) held constant over time
- Extreme events beyond normal distribution not captured (fat tails)
- Historical returns don't guarantee future performance
- No modeling of dividend payments or reinvestment timing

## Configuration Options

Edit these parameters at the top of `monte_carlo_portfolio_analysis.py`:

```python
NUM_SIMULATIONS = 10000          # More = better precision (slower)
TIME_HORIZON_YEARS = 10          # Projection period
RISK_FREE_RATE = 0.04            # Used in Sharpe ratio calculation
ENABLE_REBALANCING = False       # Set True for quarterly rebalancing
REBALANCE_FREQUENCY = 63         # Trading days between rebalances

# Monthly contributions (new!)
ENABLE_CONTRIBUTIONS = False     # Set True to add monthly contributions
MONTHLY_CONTRIBUTION = 1000      # Dollar amount added each month
CONTRIBUTION_GROWTH_RATE = 0.03  # 3% annual increase (raises/inflation)
```

**Dashboard:** All parameters can be adjusted via sliders—no code editing required!

### Enable Rebalancing

Set `ENABLE_REBALANCING = True` to automatically rebalance portfolio to target weights every quarter (63 trading days).

### Customize Correlation Assumptions

Edit `ASSET_CLASS_CORRELATIONS` dictionary to adjust how different asset classes correlate. For example:
```python
('us_equity', 'bonds'): 0.15  # Low correlation = good diversification
('us_equity_tech', 'us_equity_tech'): 1.00  # Same class = high correlation
```

The correlation matrix is built automatically from your portfolio's asset classes—no hardcoded ticker lists!

## Dashboard Features

The Streamlit dashboard (`dashboard.py`) provides:

### Interactive Controls
- **Simulation parameters**: Number of simulations, time horizon, risk-free rate
- **Monthly contributions**: Enable/disable, set amount, configure annual growth
- **Rebalancing**: Toggle quarterly rebalancing on/off

### Visualizations
1. **Distribution Tab**: Histogram of final values with percentile table
2. **Projections Tab**: Sample paths with median and percentile bands
3. **Risk Analysis Tab**: Probability metrics and drawdown distribution
4. **Portfolio Tab**: Current allocation breakdown and pie chart

### Real-Time Updates
- Adjust sliders → click "Run Simulation" → see results instantly
- Progress bar shows simulation status
- All metrics update dynamically

## Example Use Cases

### Retirement Planning
```python
# In monte_carlo_portfolio_analysis.py
TIME_HORIZON_YEARS = 30
ENABLE_CONTRIBUTIONS = True
MONTHLY_CONTRIBUTION = 1500
CONTRIBUTION_GROWTH_RATE = 0.03  # Annual raises
```

### Aggressive Accumulation
```python
TIME_HORIZON_YEARS = 10
ENABLE_CONTRIBUTIONS = True
MONTHLY_CONTRIBUTION = 3000
ENABLE_REBALANCING = True  # Maintain target allocation
```

### Inheritance/Windfall (No Contributions)
```python
ENABLE_CONTRIBUTIONS = False  # Just let it grow
TIME_HORIZON_YEARS = 20
ENABLE_REBALANCING = True
```

---

