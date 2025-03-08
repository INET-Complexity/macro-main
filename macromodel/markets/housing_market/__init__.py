"""Housing market simulation package.

This package implements a comprehensive housing market model for economic
simulations. It provides functionality for simulating property transactions,
rental agreements, and market dynamics in an agent-based framework.

Package Structure:
1. Core Components:
   - housing_market.py: Main market implementation
   - housing_market_ts.py: Time series tracking
   - func/: Market functions and algorithms

Key Features:
1. Market Operations:
   - Property sales and purchases
   - Rental agreements
   - Price discovery
   - Market clearing
   - Transaction processing

2. Property Management:
   - Value tracking and updates
   - Ownership records
   - Occupancy status
   - Property characteristics

3. Market Analysis:
   - Price trends
   - Transaction volumes
   - Market composition
   - Efficiency metrics

4. Time Series Data:
   - Historical tracking
   - Market indicators
   - Statistical analysis
   - Performance metrics

The package integrates with the broader economic simulation by:
- Processing mortgage applications
- Tracking household wealth
- Managing property ownership
- Recording market statistics

Example Usage:
    from macromodel.markets.housing_market import HousingMarket

    # Create a new housing market
    market = HousingMarket.from_data(
        country_name="USA",
        scale=1000,
        data=market_data,
        config=market_config
    )

    # Run market operations
    market.update_property_value()
    market.clear(status, prices, rents)
    market.process_housing_market_clearing(states, mortgages, wealth)
"""

from macromodel.markets.housing_market.housing_market import HousingMarket
from macromodel.markets.housing_market.housing_market_ts import create_housing_market_timeseries

__all__ = ["HousingMarket", "create_housing_market_timeseries"]
