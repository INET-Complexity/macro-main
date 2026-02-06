# SyntheticGoodsMarket

The `SyntheticGoodsMarket` class is a container for preprocessed goods market data that organizes market relationships, trade patterns, and exchange rates for model initialization.

## Core Functionality

The class handles:

1. Exchange Rate Processing:
    - Historical exchange rate data
    - Inflation rate relationships
    - Growth rate correlations
    - Rate prediction model estimation

2. Market Data Organization:
    - Buyer-seller relationships
    - Trade flow patterns
    - Price level initialization
    - Market clearing conditions

3. Parameter Processing:
    - Exchange rate model parameters
    - Price adjustment factors
    - Trade flow coefficients
    - Growth-inflation relationships

## Key Attributes

- `country_name`: Country identifier for data collection
- `exchange_rates_model`: Preprocessed exchange rate model for initializing price dynamics, relating exchange rates to inflation and growth patterns

## Factory Methods

The class provides a factory method `from_readers` that creates a `SyntheticGoodsMarket` instance by:

1. Collecting historical exchange rates
2. Matching with inflation and growth data
3. Cleaning and aligning time series
4. Estimating exchange rate model parameters

The method takes:

- `country_name`: Country to process data for
- `year`: Base year for preprocessing
- `quarter`: Base quarter for preprocessing
- `readers`: Data source access
- `exogenous_data`: External economic data
- `max_timeframe`: Maximum historical periods (default: 40)

# Implementation

::: macro_data.processing.synthetic_goods_market.synthetic_goods_market
    options:
        members:
            - SyntheticGoodsMarket
