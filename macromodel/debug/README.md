# Debug Logging System

This module provides optional, hypothesis-specific logging for investigating simulation behavior. All logging is **opt-in** and has zero performance impact when disabled.

## Design Philosophy

1. **Parsimonious**: Each logger targets a specific hypothesis with only relevant metrics
2. **Non-invasive**: Uses posthook system, doesn't modify core simulation logic
3. **Clean separation**: Debug code isolated from production code
4. **Performance-aware**: Only captures data when explicitly enabled

## Architecture

### Posthook System

The `Simulation` class now supports `posthooks` - callbacks executed after each timestep:

```python
def my_debug_hook(simulation, t, year, month):
    """Called after each timestep with full access to simulation state."""
    # Extract and log metrics here
    pass

# Register the hook
simulation.posthooks.append(my_debug_hook)

# Run simulation - hook will be called automatically
simulation.run()
```

**Key timing**: Posthooks are called AFTER:
- All markets have cleared (labor, credit, housing, goods)
- All metrics have been updated
- All time series have been recorded

This ensures you capture the **realized state** of the economy, not intermediate values.

## Available Loggers

### H1: Labor Substitution Logger (`tfp_labor_logger.py`)

**Hypothesis**: TFP growth causes unemployment by allowing firms to produce same output with fewer workers, while demand expectations remain flat.

**Key Mechanism**:
```
TFP↑ → effective_capacity↑ → BUT demand_expectations flat
→ target_production stays same → desired_labour↓
→ unemployment↑ → wage_bill↓ → household_income↓
```

**Usage**:

```python
from macromodel.debug import TFPLaborLog, capture_tfp_labor_snapshot

# Create log object
labor_log = TFPLaborLog()

# Define hook
def log_labor_dynamics(sim, t, year, month):
    snapshot = capture_tfp_labor_snapshot(sim, t)
    labor_log.add_snapshot(snapshot)

# Register hook
simulation.posthooks.append(log_labor_dynamics)

# Run simulation
simulation.run()

# Analyze results
import pandas as pd
df = pd.DataFrame(labor_log.to_dict_list())
df.to_csv("labor_dynamics.csv")

# Check hypothesis
final_unemployment = df['unemployment_rate'].iloc[-1]
final_tfp = df['avg_tfp_multiplier'].iloc[-1]
```

**Captured Metrics**:
- `avg_tfp_multiplier`: Average TFP across all firms
- `tfp_by_industry`: TFP by industry (ndarray)
- `executed_productivity_investment`: Total TFP investment
- `total_production`, `total_target_production`: Production metrics
- `avg_capacity_utilization`: Actual/target production ratio
- `total_estimated_demand`: Firm demand expectations
- `total_estimated_growth`: Average growth estimate
- `total_desired_labour`: Labor firms want to hire
- `total_actual_labour`: Labor firms actually get
- `total_employment`: Number of employed individuals
- `unemployment_rate`: Fraction unemployed
- `labor_shortage`: Desired - actual labor (>0 = shortage)
- `total_wage_bill`, `avg_wage`: Wage metrics
- `total_household_income`: Aggregate income
- `total_inventory`: Unsold goods inventory

**Derived Metrics** (computed from snapshot):
- `labor_utilization()`: Fraction of desired labor obtained
- `demand_supply_gap()`: Demand - production (>0 = excess demand)

## Example: Complete Workflow

```python
# debugging_experiments/test_h1.py
import pandas as pd
from macromodel.debug import TFPLaborLog, capture_tfp_labor_snapshot
from macromodel.simulation import Simulation
# ... (import data, create configurations)

# Setup simulations
sim_with_tfp = Simulation.from_datawrapper(data, config_with_tfp)
sim_no_tfp = Simulation.from_datawrapper(data, config_no_tfp)

# Setup logging
log_with = TFPLaborLog()
log_without = TFPLaborLog()

sim_with_tfp.posthooks.append(
    lambda sim, t, y, m: log_with.add_snapshot(capture_tfp_labor_snapshot(sim, t))
)
sim_no_tfp.posthooks.append(
    lambda sim, t, y, m: log_without.add_snapshot(capture_tfp_labor_snapshot(sim, t))
)

# Run
sim_with_tfp.run()
sim_no_tfp.run()

# Analyze
df_with = pd.DataFrame(log_with.to_dict_list())
df_no = pd.DataFrame(log_without.to_dict_list())

# Compare unemployment trajectories
import matplotlib.pyplot as plt
plt.plot(df_with['t'], df_with['unemployment_rate'], label='With TFP')
plt.plot(df_no['t'], df_no['unemployment_rate'], label='No TFP')
plt.xlabel('Timestep')
plt.ylabel('Unemployment Rate')
plt.legend()
plt.show()

# Statistical test
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(
    df_with['unemployment_rate'],
    df_no['unemployment_rate']
)
print(f"Unemployment difference: t={t_stat:.3f}, p={p_value:.4f}")
```

## Adding New Loggers

To add logging for a new hypothesis:

### 1. Create Logger Module

Create `macromodel/debug/hypothesis_name_logger.py`:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class HypothesisSnapshot:
    """Snapshot for hypothesis X."""
    t: int
    # Add hypothesis-specific metrics
    metric1: float
    metric2: np.ndarray

@dataclass
class HypothesisLog:
    """Log container."""
    snapshots: list[HypothesisSnapshot] = field(default_factory=list)

    def add_snapshot(self, snapshot):
        self.snapshots.append(snapshot)

    def to_dict_list(self):
        return [
            {
                "t": s.t,
                "metric1": s.metric1,
                # Handle arrays appropriately
            }
            for s in self.snapshots
        ]

def capture_hypothesis_snapshot(simulation, t: int) -> HypothesisSnapshot:
    """Extract hypothesis-specific metrics from simulation."""
    country = simulation.countries["CAN"]

    # Extract only metrics relevant to your hypothesis
    metric1 = country.firms.ts.current("some_metric").sum()

    return HypothesisSnapshot(
        t=t,
        metric1=metric1,
    )
```

### 2. Register in `__init__.py`

```python
from macromodel.debug.hypothesis_name_logger import (
    HypothesisLog,
    HypothesisSnapshot,
    capture_hypothesis_snapshot,
)
```

### 3. Use in Scripts

```python
from macromodel.debug import HypothesisLog, capture_hypothesis_snapshot

log = HypothesisLog()
simulation.posthooks.append(
    lambda sim, t, y, m: log.add_snapshot(capture_hypothesis_snapshot(sim, t))
)
simulation.run()
```

## Best Practices

### DO:
- ✓ Keep loggers focused on specific hypotheses
- ✓ Only capture metrics directly relevant to hypothesis
- ✓ Use dataclasses for type safety
- ✓ Provide `to_dict_list()` for easy DataFrame conversion
- ✓ Document what mechanism you're testing
- ✓ Make logging opt-in

### DON'T:
- ✗ Capture everything "just in case" (creates performance issues)
- ✗ Modify simulation logic in loggers (read-only!)
- ✗ Use global variables (pass data through log objects)
- ✗ Make logging mandatory (always opt-in)
- ✗ Log expensive computations (keep snapshots lightweight)

## Performance Considerations

- **Without logging**: Zero overhead (posthooks list is empty)
- **With logging**: ~1-5% overhead per active logger
- **Memory**: Each snapshot ~1-10 KB depending on metrics
- **For 1000 timesteps**: ~1-10 MB per log

If performance becomes an issue:
1. Reduce snapshot frequency (log every N timesteps)
2. Use sampling instead of full arrays
3. Stream to disk instead of in-memory storage

## Troubleshooting

### AttributeError: object has no attribute 'X'

Check that you're accessing metrics AFTER they've been computed in the timestep. Posthooks run after `update_realised_metrics()`, so all `ts.current()` values should be available.

### Empty/NaN values

Ensure the metric exists in your configuration. Some metrics are only recorded when certain features are enabled.

### Memory issues with long simulations

Use sampling or disk streaming:

```python
# Sample every 10 timesteps
if t % 10 == 0:
    log.add_snapshot(capture_snapshot(sim, t))

# Or stream to CSV
with open("log.csv", "a") as f:
    if t == 0:
        f.write("t,metric1,metric2\n")  # Header
    f.write(f"{t},{metric1},{metric2}\n")
```

## Integration with reproduce_reetik.py

The main debugging script uses this system. See `debugging_experiments/reproduce_reetik.py` for a complete example with:
- Optional logging via `enable_logging` flag
- Automatic CSV export
- Diagnostic output
- Hypothesis testing

```bash
# Run with logging
cd debugging_experiments
python reproduce_reetik.py  # Logging enabled by default

# Results:
# - h1_logs_with_tfp.csv
# - h1_logs_without_tfp.csv
# - Diagnostic output in console
```
