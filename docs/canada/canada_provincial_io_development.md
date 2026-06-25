# Canada Provincial IO Development Branch

This branch prepares the public MacroABM-CA repository to read and run the updated Canada provincial IO table.

The branch does **not** commit private or large raw data files. Users still provide a local `raw_data_path`, and the provincial reader expects:

```text
raw_data_path/
  icio/
    icio_2014_can_provinces.csv
```

For the current 43-sector provincial table, that file should contain the model-ready table using the legacy filename `icio_2014_can_provinces.csv`, because the reader resolves the provincial table by that name.

## Why This Branch Exists

The public repository already supports a Canada provincial model path through `DataWrapper.from_config(...)` and `DataReaders.from_raw_data(..., use_provincial_can_reader=True)`. However, the updated provincial table exposed several assumptions that were safe for the older table but not for the current one:

- the provincial ICIO reader assumed a static sector list instead of reading the sector list from the table;
- WIOD SEA variables were mapped to IO sectors using broad first-letter fallback rules;
- capital/investment allocation could place positive capital mass into province-sector cells with effectively zero IO value added;
- zero trade-flow blocks could create `0/0 -> NaN` trade proportions;
- goods-market trade-proportion normalization could produce NaNs when a vector had zero total mass;
- local pandas/Arrow dtype behaviour could break HFCS loading before the model build reached the IO table.

The changes below are intended to make the reader path table-aware, more explicit about sector matching, and safer around sparse province-sector/trade blocks.

## Code Changes

### `macro_data/readers/default_readers.py`

The provincial ICIO reader now detects the model sector list directly from the provincial table. This avoids forcing the static `ALL_INDUSTRIES` list onto a table whose actual sector representation is different.

Why this is needed:

- the current Canada provincial table has 43 sectors;
- sector labels such as `A`, `D`, `J`, `B05a`, `B05b`, `B05c`, `C24a`, and `C24b` do not line up cleanly with the old static assumptions;
- downstream WIOD SEA matching must receive the actual IO sector list.

### `macro_data/readers/io_tables/sector_contracts.py`

Adds an explicit SEA-to-IO sector bridge.

This replaces broad first-letter fallback matching with explicit rules for known sector-system differences, including:

- `B` / `B05` into `B05a`, `B05b`, `B05c`, `B07`, `B09` where relevant;
- `C24` into `C24a`, `C24b`;
- WIOD `A01`, `A02`, `A03` into current IO `A`;
- WIOD `J58T60`, `J61`, `J62` into current IO `J`;
- residual `T` and `U` into `R_S`.

When a SEA sector must be split across multiple IO sectors, IO value-added weights are used.

Why this is needed:

- broad prefix matching can silently over-allocate SEA labour/capital variables to sectors that only share a first letter;
- explicit mapping makes mismatches auditable and easier to revise.

### `macro_data/readers/socioeconomic_data/wiod_sea_data.py`

The SEA reader now uses the explicit bridge before provincial splitting, then calls a province-aware reconciliation helper.

Why this is needed:

- WIOD SEA is national and has a different sector representation from the provincial IO table;
- after splitting national SEA values to provinces, each province-sector cell must remain consistent with IO value-added constraints;
- the model should not infer capital/labour allocations from broad sector-prefix matches.

### `macro_data/readers/socioeconomic_data/sea_io_reconciliation.py`

Adds province-aware SEA/IO reconciliation and an active-VA eligibility flag.

Key rule:

- IO value added remains the hard province-sector target;
- province-sector cells with effectively zero IO value added are marked inactive for capital allocation;
- labour compensation, capital compensation, and capital stock are distributed only across active cells.

Why this is needed:

- the updated table can have sparse province-sector cells;
- national SEA capital structure should not be pushed into province-sector cells where IO value added is effectively zero;
- otherwise the investment matrix can trip capital-ratio assertions.

### `macro_data/readers/icio_sea_matching.py`

Investment allocation now reads the active-VA eligibility mask produced by reconciliation.

Why this is needed:

- prevents capital/investment mass from being reintroduced into inactive province-sector cells;
- avoids capital-ratio failures caused by positive investment allocation into near-zero-VA cells.

### `macro_data/readers/io_tables/icio_reader.py`

Trade-share construction now uses safe division when total imports or exports are zero.

Why this is needed:

- sparse trade-flow blocks can otherwise produce `NaN` shares through `0/0`;
- those NaNs later propagate into market-clearing arrays.

### `macromodel/markets/goods_market/func/lib_water_bucket.py`

Goods-market trade-proportion normalization now handles zero-sum vectors safely.

Why this is needed:

- even with safer ICIO shares, some seller/buyer trade-proportion arrays can have zero total mass for a sector;
- normalization should return zeros, not NaNs.

### `macro_data/readers/population_data/hfcs_reader.py`

HFCS loading avoids pandas/Arrow string-column mutation issues by reading as object dtype and assigning converted numeric columns explicitly.

Why this is needed:

- with the current local pandas/pyarrow stack, the previous HFCS load path could fail before the model build reached the IO table.


## Successful Growth Run Preset

The positive-growth run used:

- simulation seed: `1`;
- `t_max = 41`;
- 42 saved records;
- 4-month timestep in the reference run;
- `SimpleTFPGrowth`;
- `SimpleProductivityInvestmentPlanner`;
- `SimpleTechnicalGrowth`.

Key firm settings:

```python
firms.functions.productivity_growth.name = "SimpleTFPGrowth"
firms.functions.productivity_growth.parameters = {"investment_effectiveness": 0.3}

firms.functions.productivity_investment_planner.name = "SimpleProductivityInvestmentPlanner"
firms.functions.productivity_investment_planner.parameters = {
    "n_firms": n_industries,
    "hurdle_rate": 0.01,
    "max_investment_fraction": 0.2,
    "investment_effectiveness": 0.3,
    "investment_elasticity": 0.3,
    "tfp_investment_share": 0.5,
    "technical_investment_effectiveness": 0.3,
    "technical_diminishing_returns": 0.1,
    "price_weight": 0.4,
    "usage_weight": 0.3,
    "potential_weight": 0.3,
}

firms.functions.technical_coefficients_growth.name = "SimpleTechnicalGrowth"
firms.functions.technical_coefficients_growth.parameters = {
    "investment_effectiveness": 0.3,
    "diminishing_returns_factor": 0.1,
}

firms.parameters.tfp_base_growth_rate = 0.001
firms.parameters.tfp_investment_elasticity = 0.5
firms.parameters.max_productivity_investment_fraction = 0.15
firms.parameters.max_productivity_cash_fraction = 0.3
firms.parameters.tfp_investment_share = 0.4
firms.parameters.technical_investment_effectiveness = 0.15
firms.parameters.technical_diminishing_returns = 0.5
```

This preset is documented as a tested run configuration, not proposed as a silent default change.