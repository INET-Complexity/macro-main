# HFCS Data Exploration and Spoofing Strategy

## Overview

This document describes the structure of the HFCS (Household Finance and Consumption Survey) data and outlines a strategy for creating spoofed versions that maintain statistical properties while protecting confidentiality.

## Data Structure

### Files
The HFCS data consists of three main files:

1. **P1.csv** (Personal/Individual data): 28,845 individuals
2. **H1.csv** (Household data): 12,035 households
3. **D1.csv** (Derived data): 12,035 households

### Key Relationships

```
P1 (individuals)
  └── hid (household ID) → H1.id (household)
                           └── H1.id = D1.ID (derived household data)
```

**Critical invariants:**
- Each individual in P1 has a `hid` that must exist in H1
- H1 and D1 must have identical household IDs
- Household sizes range from 1 to 16 individuals (mean: 2.4)
- All files share common metadata columns: `SA0100`, `SA0010`, `survey`, `IM0100`

## Column Structure Analysis

### P1.csv (Personal Data)
- **Total columns:** 129
- **ID columns:** `id` (individual), `hid` (household)
- **Categorical columns:** 17 (e.g., gender, education, employment status)
- **Numerical columns:** 49 (e.g., age, income, benefits)
- **Flag columns:** 60 (columns starting with 'f' that flag data quality/imputation)

**Key columns:**
- `SA0100`: Country code (only 'FR' in this dataset)
- `RA0300`: Age (range: 0-85, mean: 40.9)
- `RA0200`: Gender (values: 1, 2)
- `PG0110`: Employee Income (range: 10-4,330,000, highly skewed, 58.75% missing)
- `PE0100a-i`: Employment status indicators
- `PA0200`: Education level

### H1.csv (Household Data)
- **Total columns:** 920
- **ID columns:** `id` (household)
- **Categorical columns:** 185
- **Numerical columns:** 281
- **Flag columns:** 453

**Key columns:**
- `HW0010`: Survey weight (range: 1.34-22,979.26, mean: 2,411.11) - **CRITICAL**
- `HB0300`: Tenure status of main residence
- `HB2300`: Rent paid
- `HB2410`: Number of other properties
- `HI0220`: Amount spent on consumption

**Asset columns (HB/DA series):**
- `HB1101-1103`: Property values
- `HB1131a-c`: Vehicle values
- Financial assets across multiple columns

**Liability columns (HC/DL series):**
- Mortgage balances
- Credit lines
- Credit card debt
- Other loans

### D1.csv (Derived Data)
- **Total columns:** 208
- **ID columns:** `ID` (household)
- **Categorical columns:** 4
- **Numerical columns:** 203
- **No flag columns** (these are computed values)

**Key derived columns:**
- `DA1110`: Value of main residence (0-6,753,340)
- `DA1120`: Value of other properties
- `DA2101-2109`: Financial asset categories
- `DL1110`: Outstanding mortgage balance
- `DI2000`: Total income (-162,010 to 4,529,400) - **can be negative!**
- `DN3001`: Net wealth
- Various ratio indicators (e.g., `DODARATIO`, `DOLTVRATIO`)

## Missing Data Patterns

### P1.csv
- 56 columns have missing data
- Employment income (`PG0110`): 58.75% missing
- Self-employment income: high missingness
- Most personal characteristic columns have low missingness

### H1.csv
- 431 columns have missing data
- High missingness is expected for asset/liability columns (not all households have all asset types)
- Pattern: if household doesn't have an asset type, related columns are missing

### D1.csv
- 88 columns have missing data
- Missing data correlates with H1 missing data
- Derived columns inherit missingness from source columns

## Data Coherence Requirements

Based on [validate_hfcs_coherence.py](validate_hfcs_coherence.py), spoofed data must pass these checks:

### ✓ ID Relationships
- Individual IDs in P1 must be unique
- All household IDs in P1.hid must exist in H1.id
- H1.id and D1.ID must be identical sets

### ✓ Country Consistency
- SA0100 (country code) must be consistent across all files
- No missing country codes allowed

### ✓ Survey Consistency
- Survey column must be consistent across files

### ✓ Household Aggregation
- Each household in H1 should have at least one individual in P1
- Household sizes should be reasonable (1-20 individuals)

### ✓ Survey Weights
- HW0010 must be positive and consistent between H1 and D1
- Weights are critical for statistical inference - **preserve or regenerate carefully**

### ✓ Common Columns
- SA0010, SA0100, IM0100, survey must match between H1 and D1

### ✓ Demographic Consistency
- Ages (RA0300) must be non-negative and reasonable (0-120)
- Gender codes must be valid (1, 2)

## Spoofing Strategy

### Phase 1: Preserve Structure
Keep these columns as-is or with minimal modification:

1. **Country identifier**: `SA0100` = 'FR' (preserve)
2. **Survey metadata**: `SA0010`, `survey`, `IM0100` (preserve)
3. **Survey weights**: `HW0010` (preserve - critical for analysis)
4. **ID structure**: Maintain number of households and individuals per household

### Phase 2: Spoof by Column Type

#### Type A: Low-Cardinality Categorical (Resample)
**Strategy:** Randomly sample from observed distribution

**Columns:**
- Gender (`RA0200`): Sample from {1, 2} with observed frequencies
- Education levels (`PA0200`)
- Employment indicators (`PE0100a-i`)
- Tenure status (`HB0300`)
- Binary yes/no indicators

**Implementation:**
```python
# Example: Resample gender maintaining distribution
original_dist = df['RA0200'].value_counts(normalize=True)
df['RA0200'] = np.random.choice(
    original_dist.index,
    size=len(df),
    p=original_dist.values
)
```

#### Type B: Continuous Numerical (Fit Distribution)
**Strategy:** Fit parametric or non-parametric distribution and sample

**Considerations:**
- Check if strictly positive (e.g., asset values)
- Check if can be negative (e.g., income with losses, `DI2000`)
- Check if has structural zeros (e.g., many households have 0 for certain assets)
- Preserve missingness patterns

**Columns:**
- Age (`RA0300`): Truncated at [0, 120]
- Income columns (`PG0110`, `DI2000`, etc.)
- Asset values (`DA1110`, `DA1120`, etc.)
- Liability values (`DL1110`, `DL1120`, etc.)

**Implementation approaches:**

1. **For strictly positive values with structural zeros:**
```python
# Two-stage model: Bernoulli for having asset + Lognormal for amount
has_asset_prob = (df['DA1110'] > 0).mean()
has_asset = np.random.random(len(df)) < has_asset_prob

positive_values = df.loc[df['DA1110'] > 0, 'DA1110']
log_params = positive_values.apply(np.log).agg(['mean', 'std'])

new_values = np.zeros(len(df))
new_values[has_asset] = np.random.lognormal(
    mean=log_params['mean'],
    sigma=log_params['std'],
    size=has_asset.sum()
)
df['DA1110'] = new_values
```

2. **For values that can be negative:**
```python
# Use normal distribution or empirical quantiles
mean, std = df['DI2000'].mean(), df['DI2000'].std()
df['DI2000'] = np.random.normal(mean, std, len(df))
```

3. **For highly skewed distributions (alternative):**
```python
# Use quantile-based resampling to preserve distribution shape
quantiles = np.linspace(0, 1, len(df))
original_quantiles = df['PG0110'].quantile(quantiles)
np.random.shuffle(quantiles)
df['PG0110'] = original_quantiles.loc[quantiles].values
```

#### Type C: High-Cardinality Discrete (Permute)
**Strategy:** Randomly shuffle within column to break individual-level links

**Columns:** (None identified in this dataset, but could apply to specific categorical codes)

#### Type D: Flag Columns (Linked)
**Strategy:** Regenerate based on spoofed data characteristics

**Columns:**
- All columns starting with 'f' (60 in P1, 453 in H1)
- These indicate data quality, imputation flags, etc.

**Implementation:**
```python
# Example: Set flag based on whether value is missing
df['fPG0110'] = df['PG0110'].isna().astype(int)
```

### Phase 3: Regenerate D1.csv
**Strategy:** D1 contains derived/calculated variables from H1 and P1

**Approach:**
1. Spoof P1 and H1 first
2. Recompute D1 columns using the HFCS methodology (if available)
3. **OR** spoof D1 independently but ensure:
   - Same household IDs as H1
   - Same weights (HW0010) as H1
   - Same metadata columns (SA0100, etc.)
   - Preserve correlations between D1 numerical columns

**Option 1 (Simple):** Treat D1 as independent and spoof numerically
**Option 2 (Complex):** Attempt to reconstruct calculation logic

### Phase 4: Maintain Correlations

Some columns are likely correlated and spoofing independently would break realistic patterns:

**Key correlations to preserve:**
1. **Age and Income:** Older workers typically have higher income (up to a point)
2. **Age and Assets:** Older households typically have more accumulated wealth
3. **Income and Assets:** Higher income correlates with higher assets
4. **Assets and Liabilities:** Households with property have mortgages
5. **Employment and Income:** Employment status correlates with income sources

**Implementation approach:**
```python
# Use copula methods or conditional distributions
# Example: Sample income conditional on age

# 1. Bin age into groups
age_bins = pd.cut(df['RA0300'], bins=[0, 30, 45, 60, 120])

# 2. Sample income within age groups
for age_group in age_bins.unique():
    mask = age_bins == age_group
    income_dist = df.loc[mask, 'PG0110'].dropna()

    # Sample from empirical distribution within group
    n_samples = mask.sum()
    df.loc[mask, 'PG0110'] = np.random.choice(
        income_dist,
        size=n_samples,
        replace=True
    )
```

### Phase 5: Preserve Missingness Patterns

**Strategy:** Missingness is often informative (e.g., not employed → no employment income)

**Approach:**
1. Save original missingness masks for all columns
2. Spoof non-missing values
3. Reapply missingness masks
4. Check if missingness patterns are correlated (e.g., all asset columns missing together)

```python
# Save missingness
missing_mask = df['PG0110'].isna()

# Spoof non-missing values
non_missing_indices = ~missing_mask
df.loc[non_missing_indices, 'PG0110'] = spoofed_values

# Missing values remain NaN
```

## Implementation Plan

### Step 1: Create Spoofing Script Structure
```python
class HFCSSpoofing:
    def __init__(self, original_data_dir):
        # Load original data
        # Save metadata (weights, IDs, country codes)

    def spoof_p1(self):
        # Preserve: id, hid, SA0100, HW0010
        # Resample: categorical columns
        # Fit distributions: numerical columns
        # Regenerate: flag columns

    def spoof_h1(self):
        # Similar to P1

    def spoof_d1(self):
        # Option 1: Regenerate from P1/H1
        # Option 2: Spoof independently with correlation preservation

    def validate(self):
        # Run validate_hfcs_coherence.py

    def save(self, output_dir):
        # Save spoofed files
```

### Step 2: Iterative Refinement
1. Start with simple spoofing (random sampling)
2. Run validation checks
3. Add correlation preservation
4. Test that spoofed data produces similar aggregate statistics
5. Iterate until validation passes and statistical properties preserved

### Step 3: Verification
Compare original vs. spoofed data:
- Mean/median/std of key variables
- Distribution shapes (KS tests)
- Correlation matrices
- Regression coefficients (e.g., age → income relationship)

## Technical Considerations

### Random Seed
Set random seed for reproducibility:
```python
np.random.seed(42)
```

### Handling Special Values
- 'A', 'M' are coded missing values in some columns (see `hfcs_reader.py:342`)
- Need to handle these before numeric conversion

### Memory Efficiency
- H1 has 920 columns × 12,035 rows
- Consider processing in chunks if memory constrained

### Edge Cases
- Households with 1 person vs. multiple people
- Households with no assets vs. wealthy households
- Different employment types (employed, self-employed, retired, unemployed)

## Files Created

1. **[validate_hfcs_coherence.py](validate_hfcs_coherence.py)**: Validation script that checks data coherence
2. **[explore_hfcs_data.py](explore_hfcs_data.py)**: Data exploration script for analysis
3. **hfcs_exploration_results.json**: Detailed exploration results (auto-generated)
4. **hfcs_exploration.md**: This document

## Next Steps

1. ✅ Create validation checks
2. ✅ Explore data structure
3. ✅ Document findings
4. 🔄 Implement spoofing script with strategies outlined above
5. 🔄 Test on subset of data
6. 🔄 Validate spoofed data passes all checks
7. 🔄 Compare statistical properties
8. 🔄 Generate full spoofed dataset
