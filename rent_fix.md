# Rent Jump Issue - Root Cause Analysis & Fix Suggestions

## Problem Summary
Total rent paid jumps from ~0 at t=0 to a large value (worth ~2% of GDP) at t=1, causing a significant discontinuity in GDP calculations.

## Root Cause Analysis

### Data Flow Investigation

1. **Initialization (t=0)**:
   - `households.ts.rent` initialized with `data["Rent Paid"].values` from input household data
   - Location: `macromodel/agents/households/households_ts.py:121`
   - This becomes `initial_real_rent_paid` in economy initialization
   - Location: `macromodel/economy/economy.py:208`

2. **First Simulation Step (t=1)**:
   - `households.compute_rent()` method called from housing market interactions
   - Location: `macromodel/agents/households/households.py:839`
   - Uses `housing_data.loc[house_id, "Rent"]` values for both:
     - Renting households: Line 888-892
     - Imputed rent for owning households: Line 896-900

3. **The Discontinuity**:
   - **t=0**: Uses `data["Rent Paid"]` from household survey data (likely zeros/very low)
   - **t=1**: Uses `housing_data["Rent"]` from housing market data (realistic market rents)
   - These two data sources are inconsistent, creating the jump

## Key Files to Debug

### Primary Investigation Points
1. **Household Data Source**: 
   - File: Input data used in `DataWrapper` (likely `data.pkl`)
   - Check: `data["Rent Paid"]` column values
   - Expected: Should be close to zero or very low

2. **Housing Market Data**:
   - Variable: `housing_data` DataFrame passed to `households.compute_rent()`
   - Check: `housing_data["Rent"]` column values
   - Expected: Should be realistic market rent values

3. **Data Initialization**:
   - File: `macromodel/agents/households/households_ts.py:121`
   - Check: How `data["Rent Paid"].values` is processed during initialization

### Debugging Breakpoints

Set breakpoints at these locations to inspect data:

1. **Initial Rent Values**:
   ```python
   # File: macromodel/agents/households/households_ts.py:121
   rent=data["Rent Paid"].values,  # <- Break here, inspect data["Rent Paid"]
   ```

2. **Economy Initialization**:
   ```python
   # File: macromodel/economy/economy.py:208
   initial_real_rent_paid = households.ts.current("rent")  # <- Break here, check sum
   ```

3. **First Rent Computation**:
   ```python
   # File: macromodel/agents/households/households.py:888-892
   rent = housing_data.loc[
       self.states["Corresponding Inhabited House ID"][ind_renting],
       "Rent",
   ].values  # <- Break here, inspect housing_data["Rent"] and rent values
   ```

4. **Rent Aggregation**:
   ```python
   # File: macromodel/economy/economy.py:930
   self.ts.total_real_rent_paid.append([real_rent_paid.sum()])  # <- Break here, check sum
   ```

## Debugging Questions to Answer

1. **Data Source Comparison**:
   - What are the actual values in `data["Rent Paid"]` during initialization?
   - What are the actual values in `housing_data["Rent"]` during first simulation step?
   - Are these from different data sources or processing steps?

2. **Household-Housing Mapping**:
   - How are households mapped to housing units via `"Corresponding Inhabited House ID"`?
   - Are the house IDs consistent between initialization and simulation?

3. **Data Processing Chain**:
   - How is the input data processed from raw sources to `data.pkl`?
   - Is there a step where rent values get zeroed out or not properly initialized?

## Potential Fix Strategies

### Strategy 1: Fix Data Initialization (Recommended)
- Ensure `data["Rent Paid"]` reflects realistic rent values during initialization
- Modify data processing pipeline to populate rent values consistently
- Location: Data preparation/synthetic population generation

### Strategy 2: Housing Data Consistency
- Ensure `housing_data["Rent"]` values are consistent with expected initial conditions
- May require adjusting housing market initialization
- Location: Housing market setup

### Strategy 3: Gradual Transition (Fallback)
- Instead of immediate jump, gradually transition from initial to market rents over first few time steps
- Add smoothing logic in `compute_rent()` method
- Location: `macromodel/agents/households/households.py:874-902`

### Strategy 4: Initialization Override
- Override the initial rent values after household creation but before economy initialization
- Use housing market data to set consistent initial rent values
- Location: Between household creation and economy initialization

## Inspection Commands for Debugger

```python
# Check initial rent data
print("Initial rent data:", data["Rent Paid"].describe())
print("Initial rent sum:", data["Rent Paid"].sum())
print("Initial rent non-zero count:", (data["Rent Paid"] > 0).sum())

# Check housing market rent data  
print("Housing rent data:", housing_data["Rent"].describe())
print("Housing rent sum:", housing_data["Rent"].sum())

# Check household-housing mapping
print("House ID range:", self.states["Corresponding Inhabited House ID"].min(), 
      "to", self.states["Corresponding Inhabited House ID"].max())
print("Housing data index range:", housing_data.index.min(), "to", housing_data.index.max())

# Check rent calculation components
print("Households renting:", ind_renting.sum())
print("Households owning:", ind_owning.sum()) 
print("Social housing:", len(ind_social_housing))
```

## Macro_Data Package Investigation

### Data Processing Pipeline for Rent Fields

The rent initialization involves multiple stages in the `macro_data` package:

#### 1. Raw Data Source (HFCS Survey Data)
- **File**: `macro_data/readers/population_data/hfcs_reader.py`
- **Raw Field**: `"HB2300"` → mapped to `"Rent Paid"`
- **Source**: European Central Bank's Household Finance and Consumption Survey
- **Issue**: Survey data may have many zeros or unrealistic values

#### 2. Household Processing Pipeline
**File**: `macro_data/processing/synthetic_population/hfcs_household_tools.py`

```python
# Lines 221-239: Critical rent processing steps
household_data.loc[:, "Rent Paid"] *= scale  # Scale up survey data
household_data.loc[households_renting & (household_data["Rent Paid"] == 0.0), "Rent Paid"] = np.nan  # Mark zeros as missing
# Apply iterative imputer to fill missing values
household_data = apply_iterative_imputer(household_data, ["Type", "Rent Paid", "Value of the Main Residence"])
household_data.loc[household_data["Rent Paid"] < social_housing_rent, "Rent Paid"] = social_housing_rent  # Floor at social housing
household_data.loc[households_owning, "Rent Paid"] = 0.0  # Set owners' rent to zero
```

#### 3. Housing Market Data Generation
**File**: `macro_data/processing/synthetic_matching/matching_households_with_houses.py`

```python
# Lines 462-464: The critical override
synthetic_population.household_data["Rent Paid"] = 0  # ← RESET TO ZERO!
synthetic_population.household_data["Rent Paid"] = mapped_df.loc[~mapped_df["Is Owner-Occupied"], "Rent"]  # Use housing market values
```

#### 4. Housing Data Creation Pipeline
**File**: `macro_data/processing/synthetic_matching/matching_households_with_houses.py`

The `housing_data` DataFrame is created with synthetic rent values that may not match the processed HFCS data.

### Additional Debugging Points

#### A. HFCS Data Source Validation
Set breakpoints in:
```python
# File: macro_data/readers/population_data/hfcs_reader.py:95
"HB2300": "Rent Paid",  # Check if HB2300 column exists and has realistic values
```

#### B. Household Processing Steps
```python
# File: macro_data/processing/synthetic_population/hfcs_household_tools.py:221-239
# Before scaling
print("Original HFCS rent data:", household_data["Rent Paid"].describe())

# After scaling but before imputation
print("Scaled rent data:", household_data["Rent Paid"].describe())

# After imputation
print("Imputed rent data:", household_data["Rent Paid"].describe())
```

#### C. Housing Market Override
```python
# File: macro_data/processing/synthetic_matching/matching_households_with_houses.py:462-464
print("Rent paid before override:", synthetic_population.household_data["Rent Paid"].sum())
print("Housing market rent values:", mapped_df.loc[~mapped_df["Is Owner-Occupied"], "Rent"].describe())
print("Rent paid after override:", synthetic_population.household_data["Rent Paid"].sum())
```

#### D. Data Wrapper Comparison
```python
# In your debugging script
data_wrapper = DataWrapper.init_from_pickle("data.pkl")
households_data = data_wrapper.countries["FRA"].population.household_data
housing_data = data_wrapper.countries["FRA"].housing_market.housing_data

print("Household Rent Paid sum:", households_data["Rent Paid"].sum())
print("Housing Rent sum:", housing_data["Rent"].sum()) 
print("Are they using same rent values?", households_data["Rent Paid"].sum() == housing_data["Rent"].sum())
```

### Root Cause Hypothesis

The **critical issue** is at line 462 in `matching_households_with_houses.py`:

```python
synthetic_population.household_data["Rent Paid"] = 0  # ← This resets all rent to zero!
```

This happens **after** all the careful HFCS processing and imputation, effectively discarding the processed survey data and replacing it with housing market synthetic data.

### Key Questions for Debugging

1. **Why is there a complete override?** Why does line 462 reset all rent to zero?
2. **Data source mismatch?** Are `households_data["Rent Paid"]` and `housing_data["Rent"]` supposed to be the same?
3. **Timing issue?** Is the housing market data not initialized when households are processed?

### Enhanced Fix Strategies

#### Strategy A: Remove the Reset (Investigate First)
- Investigate why line 462 exists - there may be a good reason
- Check if removing the reset breaks other functionality
- May need to ensure housing market data is consistent instead

#### Strategy B: Consistent Data Generation  
- Ensure housing market rent values match processed HFCS data
- Modify housing data generation to use HFCS-derived rents as baseline

#### Strategy C: Gradual Override
- Instead of complete reset, gradually adjust HFCS values toward market values
- Preserve the processed survey data baseline

## Expected Resolution

Once the data inconsistency is identified and fixed, the rent values should:
1. Start at realistic levels at t=0 
2. Show normal market dynamics (small changes) between time steps
3. Eliminate the 2% GDP jump between t=0 and t=1

The fix will likely involve ensuring both data sources (household survey data and housing market data) are derived from the same underlying rent values or properly reconciled during data preparation.