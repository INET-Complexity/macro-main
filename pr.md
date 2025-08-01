# Bundle-based Input Substitution for Energy Industries

## Overview
This PR implements a bundle-based weighting mechanism for intermediate and capital inputs that allows for substitution between inputs within the same bundle. The implementation is particularly useful for energy industries where different energy sources (oil, coal, gas, refined petroleum products) can be substituted for one another based on relative prices and productivity/depreciation rates.

## Key Changes
- Added `BundleWeightedTargetIntermediateInputsSetter` and `BundleWeightedTargetCapitalInputsSetter` classes that extend the existing financial targeting approach
- Implemented a weighting formula that considers:
  - Input prices relative to productivity/depreciation
  - Substitution bundles (groups of inputs that can be substituted for one another)
  - A sensitivity parameter (beta) to control the degree of substitution
- Optimized the normalization calculation using `np.einsum` for better performance
- Added support for extra taxes in the weighting calculation

## Technical Details

### Weighting Mechanism
The weighting mechanism works as follows:
1. Calculates unnormalized weights using the formula: `exp(-beta / avg_price * (price + extra_taxes) / productivity)`
2. Creates a bundle matrix that identifies which inputs can be substituted for one another
3. Normalizes weights within each bundle to ensure proper substitution behavior
4. Applies the normalized weights to the base unconstrained targets

### Bundle Definition in Configuration
Bundles are defined in the configuration through the `substitution_bundles` parameter. This parameter is a list of lists, where each inner list contains the indices of industries that belong to the same substitution bundle.

#### Example Configuration
```python
# Define energy industries that can be substituted for one another
energy_bundle = [list(industries).index(ind) for ind in ["B05a", "B05b", "B05c", "C19"]]
# B05a: Mining of coal and lignite
# B05b: Extraction of crude petroleum and natural gas
# B05c: Mining of metal ores
# C19: Manufacture of coke and refined petroleum products

# Create the substitution bundles list
substitution_bundles = [energy_bundle]

# Configure the simulation with these bundles
configuration = SimulationConfiguration(
    country_configurations={
        "CAN": CountryConfiguration.n_industry_default(
            n_industries=n_industries,
            bundles=substitution_bundles,
        )
    }
)
```

#### How Bundles Are Processed
1. The configuration's `substitution_bundles` parameter is passed to the `create_bundle_matrix` function
2. This function creates a binary matrix where:
   - Each row and column represents an industry
   - A value of 1 indicates that the industries can be substituted for one another
   - A value of 0 indicates that the industries cannot be substituted
3. The bundle matrix is then used in the weighting calculation to determine which inputs can be substituted for one another

#### Bundle Matrix Creation
The bundle matrix is created by:
1. Starting with a zero matrix of size `n_industries × n_industries`
2. For each bundle in the `substitution_bundles` list:
   - Setting all elements in the submatrix defined by the bundle indices to 1
   - This creates a symmetric matrix where all industries in the same bundle can be substituted for one another

### Weighting Formula
The weighting formula is:
```
w[i,j] = exp(-beta / avg_price * (price[j] + extra_taxes[j]) / productivity[i,j])
```

Where:
- `w[i,j]` is the weight for input j used by firm i
- `beta` is a sensitivity parameter (default: 1.0)
- `avg_price` is the average price across all inputs
- `price[j]` is the price of input j
- `extra_taxes[j]` is any additional tax on input j
- `productivity[i,j]` is the productivity of input j for firm i

For capital inputs, `productivity` is replaced with `depreciation`.

### Normalization
The weights are normalized within each bundle to ensure that:
1. The sum of weights for all inputs in the same bundle equals 1
2. This allows for proper substitution behavior where firms can shift between inputs in the same bundle based on relative prices and productivity

## Testing
Added a test case (`test_canadian_disagg`) that demonstrates the functionality with Canadian energy industries (oil, coal, gas, refined petroleum products). The test verifies that:
- The bundle-based substitution mechanism works correctly
- Emissions are properly calculated for all energy sources
- The simulation runs successfully with the new implementation

## Usage Guide

### Defining Bundles
To define substitution bundles in your configuration:

1. Identify the industries that should be substitutable (e.g., different energy sources)
2. Find the indices of these industries in your industry list
3. Create a list of these indices
4. Add this list to the `substitution_bundles` parameter in your configuration

```python
# Example: Define energy and material bundles
energy_bundle = [list(industries).index(ind) for ind in ["B05a", "B05b", "B05c", "C19"]]
material_bundle = [list(industries).index(ind) for ind in ["C16", "C17", "C18"]]

# Create the substitution bundles list
substitution_bundles = [energy_bundle, material_bundle]

# Configure the simulation
configuration = SimulationConfiguration(
    country_configurations={
        "COUNTRY": CountryConfiguration.n_industry_default(
            n_industries=n_industries,
            bundles=substitution_bundles,
        )
    }
)
```

### Adjusting Substitution Sensitivity
The `beta` parameter controls how sensitive the substitution is to price and productivity differences:
- Higher values of `beta` make substitution more sensitive to price/productivity differences
- Lower values make substitution less sensitive

You can adjust this parameter when initializing the setter classes:

```python
# Example: Create a setter with higher substitution sensitivity
setter = BundleWeightedTargetIntermediateInputsSetter(
    target_intermediate_inputs_fraction=0.8,
    credit_gap_fraction=0.5,
    beta=2.0  # Higher sensitivity to price/productivity differences
)
```

## Impact
This change allows for more realistic modeling of input substitution behavior, particularly in energy-intensive industries where firms can switch between different energy sources based on relative prices and efficiency. The implementation maintains compatibility with the existing framework while adding this new capability. 