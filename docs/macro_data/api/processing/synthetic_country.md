# SyntheticCountry

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

## Overview

The `SyntheticCountry` class is responsible for creating and managing synthetic economic data for individual countries. It coordinates the creation and initialization of various economic agents and markets that make up a country's economic system.

## Components

### Economic Agents

#### Firms

The `SyntheticFirms` component manages firm-level data and behavior:

- Firm creation and initialization
- Industry classification
- Financial data
- Production relationships

[View SyntheticFirms documentation →](synthetic_firms/index.md)

#### Banks

The `SyntheticBanks` component handles banking sector data and operations:

- Bank creation and initialization
- Balance sheet data
- Lending operations
- Financial relationships

[View SyntheticBanks documentation →](synthetic_banks/index.md)

#### Population

The `SyntheticPopulation` component manages household and individual data:

- Household creation
- Income distribution
- Wealth distribution
- Demographic data

[View SyntheticPopulation documentation →](synthetic_population/index.md)

#### Central Bank

The `SyntheticCentralBank` component handles monetary policy and operations:

- Policy rate setting
- Money supply
- Banking supervision
- Financial stability

[View SyntheticCentralBank documentation →](synthetic_central_bank/index.md)

#### Central Government

The `SyntheticCentralGovernment` component manages fiscal policy and operations:

- Tax collection
- Government spending
- Public debt
- Fiscal policy

[View SyntheticCentralGovernment documentation →](synthetic_central_government/index.md)

#### Government Entities

The `SyntheticGovernmentEntities` component handles government agencies and institutions:

- Agency creation
- Budget allocation
- Service provision
- Public administration

[View SyntheticGovernmentEntities documentation →](synthetic_government_entities/index.md)

#### Rest of the World

The `SyntheticRestOfTheWorld` component manages external economic relationships:

- Trade flows
- Capital flows
- Exchange rates
- External debt

[View SyntheticRestOfTheWorld documentation →](synthetic_rest_of_the_world/index.md)

### Markets

#### Credit Market

The `SyntheticCreditMarket` component manages credit and lending operations:

- Loan creation
- Interest rates
- Credit allocation
- Risk assessment

[View SyntheticCreditMarket documentation →](synthetic_credit_market/index.md)

#### Housing Market

The `SyntheticHousingMarket` component handles housing market operations:

- Housing stock
- House prices
- Rental market
- Housing transactions

[View SyntheticHousingMarket documentation →](synthetic_housing_market/index.md)

#### Goods Market

The `SyntheticGoodsMarket` component manages goods and services market:

- Price setting
- Supply and demand
- Market clearing
- Trade flows

[View SyntheticGoodsMarket documentation →](synthetic_goods_market/index.md)

#### Matching

The `SyntheticMatching` component handles agent matching and relationships:

- Household-firm matching
- Bank-firm matching
- Labor market matching
- Service provision matching

[View SyntheticMatching documentation →](synthetic_matching/index.md)

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

## Best Practices

1. **Data Consistency**
   - Ensure all economic relationships are properly initialized
   - Validate agent-level data against aggregate statistics
   - Maintain accounting identities

2. **Configuration Management**
   - Use appropriate country-specific settings
   - Configure proxy relationships for missing data
   - Document all configuration parameters

3. **Performance Optimization**
   - Use efficient data structures
   - Implement parallel processing where possible
   - Cache intermediate results

4. **Data Quality**
   - Validate input data
   - Handle missing values appropriately
   - Document data transformations
