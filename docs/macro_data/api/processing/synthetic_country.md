# SyntheticCountry

## Overview

The `SyntheticCountry` class is responsible for creating and managing synthetic economic data for individual countries. It coordinates the creation and initialization of various economic agents and markets that make up a country's economic system.

## Components

### Economic Data

The `CountryData` component manages country-specific economic data.

[View CountryData documentation →](country_data.md)

### Economic Agents

#### Firms

The `SyntheticFirms` component manages firm-level data and behavior:

- Firm creation and initialization
- Industry classification
- Financial data
- Production relationships

[View SyntheticFirms documentation →](synthetic_firms.md)

#### Banks

The `SyntheticBanks` component handles banking sector data and operations:

- Bank creation and initialization
- Balance sheet data
- Lending operations
- Financial relationships

[View SyntheticBanks documentation →](synthetic_banks.md)

#### Population

The `SyntheticPopulation` component manages household and individual data:

- Household creation
- Income distribution
- Wealth distribution
- Demographic data

[View SyntheticPopulation documentation →](synthetic_population.md)

#### Central Bank

The `SyntheticCentralBank` component handles monetary policy and operations:

- Policy rate setting
- Money supply
- Banking supervision
- Financial stability

[View SyntheticCentralBank documentation →](synthetic_central_bank.md)

#### Central Government

The `SyntheticCentralGovernment` component manages fiscal policy and operations:

- Tax collection
- Government spending
- Public debt
- Fiscal policy

[View SyntheticCentralGovernment documentation →](synthetic_central_government.md)

#### Government Entities

The `SyntheticGovernmentEntities` component handles government agencies and institutions:

- Agency creation
- Budget allocation
- Service provision
- Public administration

[View SyntheticGovernmentEntities documentation →](synthetic_government_entities.md)

#### Rest of the World

The `SyntheticRestOfTheWorld` component manages external economic relationships:

- Trade flows
- Capital flows
- Exchange rates
- External debt

[View SyntheticRestOfTheWorld documentation →](synthetic_rest_of_the_world.md)

### Markets

#### Credit Market

The `SyntheticCreditMarket` component manages credit and lending operations:

- Loan creation
- Interest rates
- Credit allocation
- Risk assessment

[View SyntheticCreditMarket documentation →](synthetic_credit_market.md)

#### Housing Market

The `SyntheticHousingMarket` component handles housing market operations:

- Housing stock
- House prices
- Rental market
- Housing transactions

[View SyntheticHousingMarket documentation →](synthetic_housing_market.md)

#### Goods Market

The `SyntheticGoodsMarket` component manages goods and services market:

- Price setting
- Supply and demand
- Market clearing
- Trade flows

[View SyntheticGoodsMarket documentation →](synthetic_goods_market.md)

#### Matching

The `SyntheticMatching` component handles agent matching and relationships:

- Household-firm matching
- Bank-firm matching
- Labor market matching
- Service provision matching

[View SyntheticMatching documentation →](synthetic_matching.md)

## Usage Example

```python
from macro_data.processing import SyntheticCountry
from macro_data.configuration.countries import Country

# Initialize a synthetic country
country = SyntheticCountry(
    country=Country("FRA"),
    data_readers=data_readers,
    configuration=config
)

# Create the synthetic economic system
country.create()

# Access different components
firms = country.firms
banks = country.banks
population = country.population
central_bank = country.central_bank
```

## Implementation

::: macro_data.processing.synthetic_country
    options:
        members:
            - SyntheticCountry
            - create
            - create_firms
            - create_banks
            - create_population
            - create_central_bank
            - create_central_government
            - create_government_entities
            - create_rest_of_the_world
            - create_credit_market
            - create_housing_market
            - create_goods_market
            - create_matching
            - get_firm_data
            - get_bank_data
            - get_population_data
            - get_central_bank_data
            - get_central_government_data
            - get_government_entities_data
            - get_rest_of_the_world_data
            - get_credit_market_data
            - get_housing_market_data
            - get_goods_market_data
            - get_matching_data
