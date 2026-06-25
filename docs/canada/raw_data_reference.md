# Raw Data Reference — macroabm-ca

All raw data used by the macroabm-CANADA lives in `macro-main-1/raw_data/` (separate offline folder), loaded at runtime via
`macro_data.DataWrapper.from_config(raw_data_path=...)`. The sections below describe each
relevant folder and file added/ modified specifically for the Canada specific model: what it contains, where it is read in the code, and how it
influences the simulation.

---

## 1. `cims_prices/`

### `firm_prices.csv`

**What it contains**  
Quarterly exogenous price indices for nine energy and fossil-fuel sectors, indexed from the
calibration year (2014 = 1.0). Columns: `year`, then one column per sector code.

Sectors covered: `B05a` (coal), `B05b` (natural gas), `B05c` (petroleum crude), `C19`
(refined petroleum products), `D01a` (hydroelectric), `D01b` (gas-fired power), `D01c`
(steam/thermal power), `D01d` (solar), `D01e` (wind).

**How it is loaded**  
Read by `SectorExoPricesReader` (called from `DataReaders.from_raw_data()` in
`macro_data/readers/default_readers.py`) when `datapaths.firm_prices_path` is set.

**How it is used in the simulation**  
Activates `SectorExogenousPriceSetter` / `SectorExogenousROWPriceSetter` in place of the
default endogenous inflation rule. Each quarter the price setter interpolates this table to
impose an exogenous price level on the covered sectors. Use this when you want energy prices
to follow a prescribed path (e.g. CIMS model output) rather than emerging from supply-demand
dynamics.

---

## 2. `emission_factors/`

The entire directory is passed to the data readers via:
```python
emissions_fraction_path = raw_data_path / "emission_factors"
```

### Files loaded in model simulation

#### `emitting_fraction_CO2.csv`

A matrix of emission intensities for CO₂.  
**Rows** = energy carrier sectors (B05a, B05b, B05c, C19 — the four fossil-fuel inputs).  
**Columns** = all 50 industry sectors.  
Each cell is the CO₂ emitted (tCO₂e per unit of that carrier purchased) when sector *j*
buys from carrier sector *i*, scaled to the model's monetary units at the 2014 price level.
Used to compute `inputs_emissions` and `capital_emissions` for firms each quarter.

There are two variants:  
- `emitting_fraction_CO2.csv` — baseline (excludes D01b/D01c electricity)  
- `emitting_fraction_CO2 - wD01b&c.csv` — includes gas-fired and steam power sectors as
  additional emitting carriers

The active version is selected in `default_readers.py` based on configuration.

#### `emitting_fraction_CH4.csv`

Same column layout as the CO₂ matrix but a single row (`CH4`) covering all 50 sectors.
Values represent CH₄ emission intensity per unit purchased. Used to populate
`inputs_emissions_ch4` and `capital_emissions_ch4` for firms.

#### `emitting_fraction_consumption.csv`

Single row (`household`), 50 columns. Fraction of household consumption of each good that
generates CO₂ emissions (direct combustion — e.g. home heating gas, petrol). Used to compute
`consumption_emissions_by_good` for the household agent each quarter.

Note the difference from the firm matrix: households only directly emit from a few goods
(B05b, B05c = gas/oil for home heating; C19 = refined fuels; D01b/D01c = small fractions).

#### `emitting_fraction_investment.csv`

Same structure as `emitting_fraction_consumption.csv` but applied to household investment
(capital formation spending). Covers B05a, B05b, B05c, C19, D01b, D01c at their respective
fractions. Used to compute `investment_emissions_by_good`.

### File used only for validation (not loaded into the simulation)

#### `EN-GHG_EconSectByGas-CA_Emissions_2014_2023_v4.csv`

**Source**: Environment and Climate Change Canada — Official GHG Inventory, Economic Sector
by Gas (downloadable from the EC data portal).

**What it contains**  
Actual reported CO₂ and CH₄ emissions for Canada, 2014–2023, organised by economic sector
(using an ISIC-adjacent classification). Columns follow the pattern:
`2014_CO2, 2014_CH4, 2015_CO2, 2015_CH4, ...`. An `index` column carries the ISIC-style
sector code (B05b, B05c, etc.) and an `ID_name` column carries the plain-English description.

**How it is used in the notebook `macromodel-CAE2026 - exo.ipynb`**  
Loaded to create `ts_emissions_CO2_total_yr_StatsCan` and
`ts_emissions_CH4_total_yr_StatsCan` — historical benchmark arrays that mirror the shape of
the model's output arrays `(n_years, n_industries)`. These are then overlaid on simulation
plots so model trajectories can be compared against observed data.
Calibration ratio printouts (`statscan_value / model_value`) for each sector help identify
sectors where the emission factors need adjustment.

### Files not used by the current simulation

- `emitting_fraction_CO2 - original.csv` — superseded draft; kept for provenance.
- `emitting_fraction_v2.csv` — alternative calibration; not selected in the default reader.

---

## 3. `icio/`

### `sectoral_disagg_CAN_2014_v2.csv`

**What it contains**  
A full inter-industry transaction matrix for Canada, 2014, in the model's 50-sector ISIC
classification. Rows and columns are labelled with `CountryInd` (CAN or ROW) and
`industryInd` (sector code). The matrix covers CAN×CAN intermediate flows, CAN×ROW trade
flows, and final demand columns (Household Consumption, Government Consumption, Fixed Capital
Formation). Values are in CAD thousands.

**How it is loaded**  
Activated when `use_disagg_can_2014_reader=True` in the data configuration. The path is
hardcoded in `DataReaders.from_raw_data()` (line 328 of `default_readers.py`):
```python
disagg_path = raw_data_path / "icio" / "sectoral_disagg_CAN_2014_v2.csv"
```

**How it is used**  
Replaces the standard OECD ICIO table as the calibration IO matrix for Canada. Because the
standard ICIO aggregates several sectors (e.g. all electricity into one D01 row), this
Canada-specific disaggregated table splits them into D01a–D01e and B05a–B05c, enabling
sector-level emission tracking. All firm production targets, price calibration, and household
consumption weights are initialised from this table.

### `mappings.json` (located in `oecd_econ/`)

> Note: despite being listed alongside the ICIO files conceptually, this file lives at
> `raw_data/oecd_econ/mappings.json` and is read via `oecd_econ_mapping_path`.

**What it contains**  
A JSON dictionary mapping ISIC section letters to lists of 4-digit ISIC codes:
```json
{ "A": ["A01", "A03"], "B": ["B05", "B07", "B09"], "C": [...], ... }
```

**How it is used**  
Used during data loading to group sectors by section for aggregation and reporting (e.g.
GDP-by-section breakdowns, emission totals by broad sector group).

---

## 4. `policy/`

### `consumer_carbon_price_rates.csv`

**What it contains**  
Annual consumer (fuel charge) carbon tax rates in CAD per tonne CO₂e, 2014–2050.
One column per Canadian jurisdiction: `CAN` (national backstop), `CAN_AB`, `CAN_BC`,
`CAN_MB`, `CAN_NB`, `CAN_NL`, `CAN_NS`, `CAN_ON`, `CAN_PE`, `CAN_QC`, `CAN_SK`.

The national (`CAN`) schedule reflects the federal fuel charge: $0 through 2018, rising from
$20 in 2019 to $80 in 2024, then $0 from 2025 onward (reflecting the 2025 policy repeal for
the consumer charge). Some provinces (BC, Quebec) had independent provincial rates before the
federal backstop applied.

**How it is loaded**  
Read by `ConsumerCarbonCANReader` → `CarbonPrice` class. Loaded when
`datapaths.consumer_carbon_path` is set (always set in the default configuration).

**How it is used**  
When `use_consumer_carbon_reg=True` on a `CountryConfiguration`, the `CarbonPrice` object
returns the annual rate each year via `get_price()`. This enters `update_extra_taxes()` in
`country.py` as a per-unit marginal cost on household energy consumption, reducing demand for
carbon-intensive goods and feeding into both firm price-setting and household consumption
decisions.

### `output_based_price_system_rates.csv`

**What it contains**  
Annual OBPS carbon price rates ($/tCO₂e), same jurisdiction breakdown as the consumer rates.
Differs from the consumer schedule: rates continue rising post-2024 toward $170/tonne and
plateau there (reflecting the industrial carbon pricing pathway which was not repealed).

**How it is loaded**  
Read by `OBPSCANReader` as the `df_rates` component of `OBPSCANData`. Paired with the
policy values file below.

**How it is used**  
Provides the carbon price `P` in the OBPS cost formula:
```
obps_cost_i = (emissions_i - limit_i) × P
```
Used only when `use_obps_reg=True`.

### `output_based_price_system_policy_values_disagg.csv`

**What it contains**  
Sector-level OBPS parameters for the 50-sector disaggregated classification. One row per
sector, columns: `Industry`, `Industry_Name`, `standard_value` (baseline emission intensity,
tCO₂e per unit output in 2014 prices), `tightening_rate` (annual rate at which the benchmark
tightens post-2022), `reduction_factor` (fraction of the baseline intensity that sets the
initial allowance, typically 0.8).

Key parameter values:
- `reduction_factor = 0.8` for all sectors
- `tightening_rate = 0.02` for most sectors (benchmark reaches zero around 2072)
- `tightening_rate = 0.01` for B07, B09, C20, C24a, C24b (benchmark reaches zero around 2122)
- `tightening_rate = 0.00` for D01a, D01d, D01e (renewables; benchmark never tightens)

**How it is used**  
Used by `OutputBasedPriceSystemCAN.get_limit()` to compute the sector-specific emission
allowance each year. The `standard_value` is the historical reference intensity; post-2022,
the allowance declines at `tightening_rate` per year.

### `output_based_price_system_policy_values.csv`

Aggregated (non-disaggregated) version of the OBPS policy values using a cruder sector
grouping. Used when running with the standard (non-disaggregated) ICIO table.

### `output_based_price_system_policy_values_elec.csv`

Electricity-sector specific OBPS parameters (D01b gas-fired, D01c steam/thermal power).
Contains year-by-year `standard_value` and `tightening_rate` because the electricity grid
benchmark is set differently: it tracks grid emission intensity from 2021 onward and tightens
to zero by 2029–2030.

### `energy_bundle_price.csv`

**What it contains**  
Quarterly exogenous price indices (timestep, year, month) for 9 energy sectors:
`B05a, B05b, B05c, C19, D01a, D01b_1, D01b_2, D01c_1, D01c_2, D01d, D01e`.
Note that D01b and D01c each have two sub-columns (split fuel-mix variants).

Similar in purpose to `cims_prices/firm_prices.csv` but at a finer quarterly resolution and
with the electricity sub-sector split. Used when configuring `EnergyExogenousPriceSetter` or
the CIMS-linked price setters (older notebook configurations).

### `.xlsx` files

`output_based_price_system_policy_values.xlsx`, `output_based_price_system_policy_values_elec.xlsx`, `energy_bundle_price.xlsx` are the source spreadsheets from which the CSV files were generated. They are not read by the simulation.

---

## 5. `hfcs/`

The `hfcs/` folder contains four pre-processed, model-ready CSV files at its root (generated by running the HFCS processing pipeline on the raw ECB survey waves). The raw survey wave files (`hfcs/2010/`, `hfcs/2014/`, `hfcs/2017/`) are intermediate inputs to that pipeline and are not loaded directly by the simulation.

The path is set via `datapaths.hfcs_path = raw_data_path / "hfcs"` and passed to `HFCSReader.from_csv()` in `default_readers.py`.

### `New_Household_provincial.csv`

**Loaded by the simulation** (line 289 of `hfcs_reader.py`).

One row per household. Columns: `Tenure Status of the Main Residence`, `Rent Paid`, `Number of Properties other than Household Main Residence`, `Amount spent on Consumption of Goods and Services`, `Type` (household type code), income sub-components (`Rental Income from Real Estate`, `Income from Financial Assets`, `Income from Pensions`, `Regular Social Transfers`), asset values (main residence, vehicles, deposits, mutual funds, bonds, private businesses, shares, voluntary pension), liability balances (HMR mortgage, other mortgages, credit line, credit card debt, other loans), `Consumption of Consumer Goods/Services as a Share of Income`, `Corresponding Individuals ID` (list of member IDs), aggregate `Wealth`, `Income`, `Debt`, and `Weight` (cross-sectional survey weight).

The `_provincial` variant maps households to Canadian provinces (CAN_AB, CAN_BC, etc.), enabling the provincial model configuration. Feeds into `SyntheticPopulation` to initialise household agents with empirically calibrated balance sheets and consumption patterns.

### `New_Individuals_provincial.csv`

**Loaded by the simulation** (line 288 of `hfcs_reader.py`).

One row per individual. Columns: `Corresponding Household ID`, `Gender`, `Age`, `Education`, `Labour Status`, `Employment Industry` (ISIC section letter), `Employee Income`, `Self-Employment Income`, `Income from Unemployment Benefits`, `Income` (total). Used by the synthetic population pipeline to assign workers to industries and to set individual income profiles within households.

### `New_Household.csv`

Pre-processed household data without the provincial assignment step — same columns as `New_Household_provincial.csv`. Produced as an intermediate output when running the HFCS pipeline for the national (non-provincial) model configuration. Not loaded by the default reader; present for reference and for running the non-provincial Canada model.

### `New_Individuals.csv`

Pre-processed individual data without provincial assignment — same columns as `New_Individuals_provincial.csv`. Counterpart to `New_Household.csv` for the national model configuration.

---

## Notebook: `macromodel-carbon-policy-scenarios.ipynb` — relationship between EN-GHG and 3610000101 files

The notebook uses two Statistics Canada / Environment Canada datasets as **external validation
benchmarks** (not loaded by the model itself):

### `EN-GHG_EconSectByGas-CA_Emissions_2014_2023_v4.csv`

Historical GHG totals from Environment Canada's official inventory.  
The notebook loads it and maps each EC sector ID (e.g. `B05b`) to the model's
industry index, producing `ts_emissions_CO2_total_yr_StatsCan[year, industry]` — a
`(n_years, n_industries)` array of observed CO₂ and CH₄ in kt.

### `3610000101` files (Statistics Canada Supply-Use table)

Three CSV files derived from StatsCan table 36-10-0001-01 (Symmetric input-output, supply
and use, NAICS classification):

| File | Description |
|------|-------------|
| `3610000101-noSymbol.csv` | Annual nominal output by NAICS industry, 2014–2023 (Supply column) |
| `3610000101_customizedLayoutData - 2014 - processed.csv` | Full 2014 supply-use matrix in wide format (NAICS rows × NAICS columns) |
| `3610000101_customizedLayoutData - to icio_can_2014_disagg.csv` | Same matrix re-mapped to the model's 50 ISIC sectors |

**How they relate to EN-GHG**

The notebook implements a `naics_isic()` function that searches the Supply-Use
table by NAICS pattern string and aggregates production values into the model's ISIC sectors.
This produces `ts_production_nom_mu_StatsCan[year, industry]` — nominal production in CAD
for each ISIC sector.

Dividing the EN-GHG emissions by this nominal production gives an **implied emission
intensity** per dollar of output for each sector:

```
emission_intensity[sector] = EN-GHG_emissions[sector] / production_StatsCan[sector]
```

The ratio `statscan_value / model_value` is printed for each industry, directly comparing
the observed emission intensity against what the model produced in the baseline (no-policy)
scenario. This tells you whether the `emitting_fraction_CO2.csv` and
`emitting_fraction_CH4.csv` parameters are correctly calibrated: a ratio near 1.0 means the
model reproduces the observed emission intensity; values far from 1.0 indicate sectors where
the emission fractions need adjustment.

In short: **the 3610000101 files provide the denominator (production), and EN-GHG provides
the numerator (emissions), together forming the observed emission intensity benchmark used to
validate and calibrate the model's emission parameters.**
