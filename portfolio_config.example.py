"""
Example portfolio configuration - Copy this to portfolio_config.py and update with your values
"""

portfolio = {
    # Example holdings - replace with your actual portfolio
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
