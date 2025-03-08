"""Labour market simulation package.

This package implements a comprehensive labour market model for economic
simulations. It provides functionality for simulating employment relationships,
wage determination, and labour market clearing in an agent-based framework.

Package Structure:
1. Core Components:
   - labour_market.py: Main market implementation
   - labour_market_ts.py: Time series tracking
   - func/: Market functions and algorithms

Key Features:
1. Employment Management:
   - Job matching algorithms
   - Industry-specific allocation
   - Employment status tracking
   - Workforce distribution

2. Market Dynamics:
   - Hiring processes
   - Separation mechanisms
   - Industry transitions
   - Labour mobility

3. Economic Integration:
   - Wage determination
   - Labour costs
   - Industry linkages
   - Market efficiency

4. Analysis Tools:
   - Employment metrics
   - Market statistics
   - Time series tracking
   - Performance indicators

The package integrates with the broader economic simulation by:
- Managing employment relationships
- Processing wage payments
- Tracking labour market conditions
- Providing employment statistics

Example Usage:
    from macromodel.markets.labour_market import LabourMarket

    # Create a new labour market
    market = LabourMarket.from_data(
        country_name="USA",
        n_industries=10,
        initial_individual_activity=activity_data,
        initial_individual_employment_industry=industry_data,
        config=market_config
    )

    # Run market operations
    labour_costs = market.clear(firms, households, individuals)
"""

from macromodel.markets.labour_market.labour_market import LabourMarket
from macromodel.markets.labour_market.labour_market_ts import create_labour_market_timeseries

__all__ = ["LabourMarket", "create_labour_market_timeseries"]
