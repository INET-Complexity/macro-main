# SyntheticHousingMarket

The `SyntheticHousingMarket` class is a container for preprocessed housing market data that organizes housing units and their relationships with households. It handles the initial state of housing ownership and rental relationships for model initialization.

## Core Functionality

The class handles:

1. Housing Unit Data:
   - Property identification and valuation
   - Rental status and rates
   - Owner-occupancy classification
   - Property characteristics

2. Housing-Household Relationships:
   - Owner-property matching
   - Renter-property matching
   - Vacant property tracking
   - Social housing allocation

3. Market Structure:
   - Rental market availability
   - Property value distribution
   - Geographic clustering (if applicable)
   - Market segment classification

## Key Attributes

- `country_name`: Country identifier for data collection
- `housing_market_data`: DataFrame containing:
  - House ID: Unique identifier for each housing unit
  - Is Owner-Occupied: Boolean flag for owner-occupied properties
  - Corresponding Owner Household ID: ID of the owning household
  - Corresponding Inhabitant Household ID: ID of the residing household
  - Value: Property value in local currency units
  - Rent: Monthly rental amount (if applicable)
  - Up for Rent: Boolean flag for rental market availability
  - Newly on the Rental Market: Tracks new rental listings

# Implementation

::: macro_data.processing.synthetic_housing_market.synthetic_housing_market
    options:
        members:
            - SyntheticHousingMarket

# Default Implementation

The `DefaultSyntheticHousingMarket` class provides a standard implementation for processing housing market data using common data sources.

## Data Processing

The default implementation handles:

1. Data Collection:
   - Household survey data
   - Property registration records
   - Rental market statistics
   - Social housing records

2. Property Matching:
   - Owner-occupied property identification
   - Rental property classification
   - Social housing allocation
   - Vacancy tracking

3. Market Initialization:
   - Property value assignment
   - Rental rate calculation
   - Market availability flags
   - New listing identification

## Factory Methods

The class provides a factory method `init_from_datadict` that creates a `DefaultSyntheticHousingMarket` instance by:

1. Converting dictionary data to DataFrame format
2. Setting initial rental market availability
3. Establishing property-household relationships

# Implementation

::: macro_data.processing.synthetic_housing_market.default_synthetic_housing_market
    options:
        members:
            - DefaultSyntheticHousingMarket
