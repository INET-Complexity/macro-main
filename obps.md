# Output-Based Pricing System (OBPS) Implementation

This document explains the implementation of Canada's Output-Based Pricing System (OBPS) in the macroeconomic model. The changes documented here represent modifications added to the `price_competition` branch.

## What is OBPS?

The Output-Based Pricing System is a carbon pricing mechanism that sets industry-specific emission limits based on production levels. Unlike a simple carbon tax, OBPS:

- Sets emission intensity standards for high-emission industries
- Allows free allocations up to the standard (production × emission intensity × reduction factor)
- Charges carbon price only on emissions above the allocated limit
- Provides incentives for emission reductions while maintaining industrial competitiveness

## Key Files Added/Modified

### New Policy Module
- **`macromodel/policy/output_based_price_system.py`**
  - Core OBPS implementation class
  - Handles limit calculations, compliance costs, and yearly updates
  - Manages regulated industry definitions and parameters

- **`macromodel/policy/OBPS policy.csv`**
  - Industry-specific policy parameters
  - Reduction factors (0.8 = 80% free allocation)
  - Tightening rates (2% annual reduction post-2022)
  - Standard emission intensities by industry

### Configuration Changes
- **`macromodel/configurations/country_configuration.py`**
  - Added `use_obps_reg: bool = False` - Enable OBPS regulation
  - Added `use_consumer_carbon_reg: bool = False` - Consumer carbon taxes
  - Carbon pricing parameters (delays, growth rates, price levels)

### Agent Modifications
- **`macromodel/agents/firms/firms.py`** (Major changes)
  - Added `extra_marginal_taxes` parameter to all cost and price calculation methods
  - Modified demand calculations for intermediate and capital inputs to include OBPS costs
  - Enhanced `compute_taxes_paid_on_production()` to include emission-based taxes
  - Updated equity calculations to reflect OBPS cost impacts
  - Removed emission zeroing for refining firms (C19 sector)
  - Fixed industry indexing to use `get_loc()` instead of `index()`

- **`macromodel/agents/firms/func/prices.py`**
  - Enhanced price setting functions to include `extra_marginal_taxes`
  - Modified `DefaultPriceSetter` to incorporate OBPS costs in price calculations

- **`macromodel/agents/firms/func/target_*.py`** (Multiple files)
  - Updated target input calculations to account for emission taxes
  - Modified financial constraint calculations to include OBPS costs

- **`macromodel/agents/central_government/central_government.py`**
  - Added emission tracking flags (`use_emissions`, `use_obps_reg`, `use_consumer_carbon_reg`)
  - New method parameters for emission tax collection
  - Enhanced revenue calculation to include carbon tax streams
  - Added `carbon_tax_revenue_dict()` method for revenue reporting

- **`macromodel/agents/central_government/central_government_ts.py`**
  - New time series for emission tax revenues:
    - `tax_hh_consumption_emissions`
    - `tax_hh_investment_emissions` 
    - `tax_firm_input_emissions`
    - `tax_firm_capital_emissions`
    - `tax_firm_obps`

## How OBPS Works in the Model

### 1. Baseline Establishment (2017-2019)
```python
# During reference period, track actual performance
self.reference_emission += (input_emissions + capital_emissions)
self.reference_production += production

# Calculate baseline intensity at end of period (2019)
self.reference_emission_intensity = self.reference_emission / self.reference_production
```

### 2. Regulated Industries
The following ISIC industry codes are subject to OBPS regulation:
- **B05a-c**: Mining and oil/gas extraction
- **B07**: Mining and quarrying (cement, lime)
- **C16**: Wood products
- **C17**: Pulp and paper
- **C19**: Refined petroleum products
- **C20**: Chemicals and fertilizers
- **C22**: Rubber products (tires)
- **C24a-b**: Iron, steel, and aluminum
- **C29**: Automotive manufacturing

### 3. Limit Calculation

**Pre-2023 (Simple reduction factor):**
```python
limit = production × (reduction_factor × reference_emission_intensity)
# Where reduction_factor = 0.8 (80% free allocation)
```

**Post-2022 (With tightening rates):**
```python
limit = production × (standard - (standard × tightening_rate × (current_year - 2022)))
# Where tightening_rate = 0.02 (2% annual reduction)
```

### 4. Compliance Cost Calculation
```python
excess_emissions = (input_emissions + capital_emissions) - limit
obps_cost = max(0, excess_emissions) × carbon_price
```

### 5. Economic Integration

**Firm Cost Structure:**
- OBPS costs added to production taxes: `macromodel/agents/firms/firms.py:1549`
- All price calculations include emission penalties
- Input demand functions account for carbon costs

**Substitution Effects:**
- Bundle matrix enables switching between energy inputs
- Firms can substitute toward cleaner production methods
- Higher emission factors increase relative input costs

**Government Revenue:**
- OBPS payments tracked separately from carbon taxes
- Revenue flows to central government budget
- Used for fiscal policy and public spending

## Key Implementation Details

### Time Progression
1. **2017-2018**: Data collection period, no costs imposed
2. **2019**: Baseline intensity calculations finalized
3. **2019-2022**: Static limits based on reduction factors
4. **2023+**: Progressive tightening of standards

### Cost Pass-Through
OBPS costs are integrated throughout the economic system:
- **Production costs**: Direct impact on firm margins
- **Price setting**: Emission penalties reflected in output prices
- **Input demands**: Higher costs reduce demand for emission-intensive inputs
- **Investment**: Capital goods subject to emission taxes

### Revenue Recycling
Government collects OBPS revenues and can:
- Reduce other taxes
- Increase public spending
- Provide targeted support to affected industries

## Configuration Example

```python
# Enable OBPS in country configuration
country_config = CountryConfiguration(
    use_obps_reg=True,  # Enable OBPS regulation
    use_consumer_carbon_reg=False,  # Consumer carbon taxes
    carbon_tax_start_delay=40,  # Delay before carbon pricing starts
    carbon_tax_start_price=236,  # Initial carbon price ($/tCO2)
    carbon_tax_price_growth_rate=0.05,  # 5% annual price growth
    carbon_tax_price_growth_duration=8,  # Years of price growth
)
```

## Model Validation

The implementation includes debugging and validation features:
- Emission limit tracking by industry (`test_limit` array)
- Compliance cost breakdowns
- Revenue verification against government budget
- Industry-specific impact analysis

## Policy Analysis Capabilities

The OBPS implementation enables analysis of:
- **Industrial competitiveness**: Impact on different sectors
- **Emission reductions**: Effectiveness of output-based approach
- **Revenue generation**: Government income from carbon pricing
- **Economic efficiency**: Costs vs. emission reduction benefits
- **Substitution patterns**: How firms adapt production methods

## References

The implementation follows Canada's federal OBPS regulations:
- [Output-Based Pricing System Regulations](https://laws-lois.justice.gc.ca/eng/regulations/SOR-2019-266/)
- Industry coverage and compliance requirements
- Emission intensity benchmarks and tightening schedules