# Internal Documentation: Current System Analysis

## Core Components

### 1. DataWrapper
- Main container class that manages synthetic economic data for multiple countries
- Key attributes:
  - `synthetic_countries`: Dict mapping country codes to SyntheticCountry objects
  - `synthetic_rest_of_the_world`: Synthetic data for rest of world
  - `exchange_rates`: Exchange rates between countries
  - `origin_trade_proportions`: Trade proportions by origin country
  - `destination_trade_proportions`: Trade proportions by destination country
  - `configuration`: DataConfiguration settings
  - `calibration_data`: Data for model calibration
  - `industries`: List of industry codes
  - `emission_factors`: Emission factors by type
  - `emissions_energy_factors`: Energy-related emission factors

### 2. SyntheticCountry
- Container for all synthetic economic data within a country
- Key components:
  - Population (households, individuals)
  - Firms (by industry)
  - Credit Market
  - Banks
  - Central Bank
  - Central Government
  - Government Entities
  - Housing Market
  - Goods Market
  - Tax Data
  - Industry Data
  - Emission Factors

### 3. Data Creation Process
1. Configuration setup:
   - Define countries
   - Set industry aggregation level
   - Configure country-specific settings
   - Set proxy relationships for non-EU countries

2. Data Readers initialization:
   - Load raw data from various sources
   - Process industry data
   - Handle exchange rates
   - Process emissions data
   - Load tax information

3. Synthetic Country Creation:
   - For EU countries: Direct creation using EU data
   - For non-EU countries: Proxy-based creation using EU country data
   - Key steps:
     - Initialize government institutions
     - Set up financial system
     - Create population
     - Create firms
     - Set up markets
     - Match agents (households-firms-banks)
     - Initialize wealth and credit

4. Market Initialization:
   - Credit market setup
   - Housing market setup
   - Goods market setup
   - Financial relationships between agents

### 4. Key Data Dependencies
1. Country-level data:
   - Exchange rates
   - Tax rates
   - Interest rates
   - Inflation data
   - Population statistics
   - Industry structure

2. Industry-level data:
   - Input-output tables
   - Trade flows
   - Production data
   - Employment data

3. Agent-level data:
   - Household surveys
   - Firm financial data
   - Bank balance sheets
   - Government accounts

### 5. Data Coherence Requirements
1. Financial flows must balance:
   - Credit market relationships
   - Bank deposits and loans
   - Government revenues and expenditures
   - Trade flows

2. Economic indicators must be consistent:
   - GDP calculations (output, expenditure, income approaches)
   - Employment levels
   - Production values
   - Trade balances

### 6. Current Limitations
1. Country-centric design:
   - Exchange rates only at country level
   - Tax rates only at country level
   - Monetary policy only at country level
   - Trade data at country level

2. Data granularity:
   - Industry data at country level
   - Population data at country level
   - Financial data at country level

### 7. Critical Dependencies
1. Data Readers:
   - ICIO (Inter-Country Input-Output)
   - Eurostat
   - World Bank
   - OECD
   - IMF
   - Emissions data

2. Configuration:
   - Country settings
   - Industry settings
   - Agent settings
   - Market settings

3. Data Processing:
   - Industry aggregation
   - Population scaling
   - Financial flow balancing
   - Market initialization 