# MacroABM-CA Onboarding Guide

This guide is a practical orientation for groups who want to run, inspect, and
interpret the current Canada MacroABM workflow. It is intentionally
code-referenced: the goal is not only to explain what the model does, but to show where the
main pieces live and how data moves from raw files into simulated agents, markets, and
results.

> **Experimental status.** The current Canada workflow should be treated as an
> experimental research model. It is runnable, inspectable, and useful for diagnosis and
> scenario prototyping, but the data pipeline, sector matching, run presets, and
> validation workflow are still being actively refined. Outputs should be interpreted as
> beta-run evidence, not as a finalized production baseline.

Public repository:

- [`uvic-sesit/macroabm-ca`](https://github.com/uvic-sesit/macroabm-ca)

Main-branch documentation:

- [Project documentation index](https://github.com/uvic-sesit/macroabm-ca/blob/main/docs/index.md)
- [Installation guide](https://github.com/uvic-sesit/macroabm-ca/blob/main/docs/getting_started/installation.md)
- [Quickstart guide](https://github.com/uvic-sesit/macroabm-ca/blob/main/docs/getting_started/quickstart.md)

The examples below use repository-relative paths. Raw data, generated pickles, H5 outputs,
and validation artifacts are intentionally treated as local files outside the committed
public repository unless explicitly noted.

---

## 0. Suggested Starting Point

This guide is the conceptual map. The notebooks are the shortest path to actually running
the current Canada provincial model.

| Start here | Use it for | What it produces |
|------------|------------|------------------|
| `Sample_macroabm-Canada_provincial_run.ipynb` | One basic end-to-end provincial run | A local `DataWrapper` pickle, one H5 result file, and basic GDP/production plots |
| `Sample_macroabm-CANADA_run_time_iteration.ipynb` | Scenario comparison | In-memory scenario results and comparison plots for GDP, emissions, and prices |

Recommended order for beta testers:

1. Read Sections 1-6 of this guide to understand the data/model contract.
2. Run `Sample_macroabm-Canada_provincial_run.ipynb` to confirm the provincial model can
   build, run, save, and plot results on your machine.
3. Run `Sample_macroabm-CANADA_run_time_iteration.ipynb` if you want to compare scenario
   behaviour across price-setting assumptions or repeated random-seed trials.
4. Use the deeper documentation links in each section below when you need API-level or
   implementation detail.

The notebooks intentionally keep explanations short. This guide provides the model context
behind them.

### Notebook Settings And Local Outputs

The sample notebooks are runnable demonstrations, not canonical validation runs. In the
current branch they use a short quarterly-style demo setup, including `time_unit = 3`,
`seed = 0`, and a limited timestep horizon. Validation and policy runs may use different
time units, seeds, productivity settings, and horizons, so do not compare notebook outputs
to other results unless those settings are aligned.

The basic provincial notebook writes a local `DataWrapper` pickle and an H5 result file.
The scenario comparison notebook is lighter: it compares behaviour across scenario/trial
settings and mainly keeps result arrays and plots in memory. Here, a "trial" means one full
model run, usually with a distinct random seed; `n_trials = 1` means the reported path is a
single run, not an average over repeated runs.

Generated notebook outputs, pickles, H5 files, plots, and validation folders can be large
and should stay local unless a team explicitly decides to share them. H5 files from even
short provincial runs can reach multiple gigabytes.

---

## 1. What MacroABM-CA Is

MacroABM-CA is the Canada-specific provincial version of the broader MacroABM framework. The
model combines:

- a data preprocessing stack in `macro_data`
- a dynamic simulation engine in `macromodel`
- Canada-specific raw data, IO tables, policy inputs, and validation scripts

Conceptually, the current Canada beta model starts with a 2014 economic structure and builds
ten provincial economies:

- `CAN_AB`
- `CAN_BC`
- `CAN_MB`
- `CAN_NB`
- `CAN_NL`
- `CAN_NS`
- `CAN_ON`
- `CAN_PE`
- `CAN_QC`
- `CAN_SK`

Each province is represented as an interacting economy with firms, households, banks,
government entities, markets, and sectoral production structures. The provinces are linked
through a provincial IO-style trade and production network, plus a simplified `ROW` external
sector.

---

## 2. Main Repository Areas

### `macro_data/`

This package turns raw datasets into the model-ready `DataWrapper` object. It handles:

- raw-data readers
- IO table loading and transformation
- WIOD SEA / socioeconomic matching
- synthetic firms, households, banks, government entities
- trade proportions and exogenous data
- final pickled data object used by the simulation

Key entry point:

- [`macro_data/data_wrapper.py`](../macro_data/data_wrapper.py)

Further documentation:

- [Macro data overview](https://github.com/uvic-sesit/macroabm-ca/blob/main/docs/macro_data/index.md)
- [DataWrapper API documentation](https://github.com/uvic-sesit/macroabm-ca/blob/main/docs/macro_data/api/data_wrapper.md)
- [Reader API documentation](https://github.com/uvic-sesit/macroabm-ca/blob/main/docs/macro_data/api/readers/index.md)
- synthetic data processing pages under `docs/macro_data/api/processing/`

### `macromodel/`

This package is the dynamic simulation engine. It handles:

- countries / provincial economies
- firms, households, banks, governments
- goods, labour, credit, housing, and trade markets
- simulation time stepping
- H5 result writing

Key entry point:

- [`macromodel/simulation.py`](../macromodel/simulation.py)

Further documentation:

- [Macromodel overview](https://github.com/uvic-sesit/macroabm-ca/blob/main/docs/macromodel/index.md)
- [Macromodel API documentation](https://github.com/uvic-sesit/macroabm-ca/blob/main/docs/macromodel/api/index.md)
- [Agent API documentation](https://github.com/uvic-sesit/macroabm-ca/blob/main/docs/macromodel/api/agents/index.md)
- market pages under `docs/macromodel/api/markets/`

### Run Scripts And Presets

The public code path for building the provincial model starts from:

- [`macro_data/data_wrapper.py`](../macro_data/data_wrapper.py)
- [`macro_data/readers/default_readers.py`](../macro_data/readers/default_readers.py)
- [`macromodel/simulation.py`](../macromodel/simulation.py)

Project teams may keep local run scripts around these entry points. Those scripts should set
the raw data path, build a `DataWrapper` pickle, configure the provincial simulation, and
save H5 outputs.

Runnable examples:

- see the two notebooks listed in [Section 0](#0-suggested-starting-point)

### External Raw Data Folder

The public repository does not commit raw data. Users provide a local raw-data folder and
pass it as `raw_data_path` to `DataWrapper.from_config(...)`.

Common convention:

```text
path/to/raw_data/
```

Important subfolders include:

- `icio/`
- `wiod_sea/`
- `exchange_rates/`
- `emission_factors/`
- `policy/`
- household and population inputs used by HFCS readers

### Local Pickle And Output Folders

Generated model artifacts should be kept outside committed source control. Typical local
outputs include:

```text
path/to/outputs/disagg_sectorprovs_2014.pkl
path/to/outputs/results_provinces_current_table.h5
```

Longer policy runs may also create CSV summaries, plots, assumptions files, and run-status
files. These are local run products, not required source files.

---

## 3. The Current Provincial Workflow

The current beta workflow has two stages. In the public repository, these stages are exposed
through the core `DataWrapper` and `Simulation` APIs. Teams can wrap these APIs in their own
local run scripts.

### Stage A: Build the data pickle

The build step:

1. creates a default data configuration for Canada
2. enables Canada disaggregation
3. defines the ten Canadian provinces
4. sets the aggregation structure `{CAN: provinces}`
5. calls `DataWrapper.from_config(...)`
6. writes a local `DataWrapper` pickle

Code reference:

```python
data_config = configuration_utils.default_data_configuration(
    countries=["CAN"],
    aggregate_industries=False,
    proxy_country_dict={"CAN": "FRA"},
)

data_config.can_disaggregation = True
data_config.aggregation_structure = {CountryCode("CAN"): provinces}

creator = DataWrapper.from_config(
    configuration=data_config,
    raw_data_path=Path("path/to/raw_data"),
    single_hfcs_survey=True,
)

creator.save(Path("path/to/outputs/disagg_sectorprovs_2014.pkl"))
```

Source:

- [`macro_data/data_wrapper.py`](../macro_data/data_wrapper.py)
- [`macro_data/readers/default_readers.py`](../macro_data/readers/default_readers.py)

Notebook reference: see the basic provincial run notebook listed in [Section 0](#0-suggested-starting-point).

### Stage B: Run the model

The model run step does not read raw IO or SEA data directly. It loads the existing pickle:

```python
data = DataWrapper.init_from_pickle(Path("path/to/outputs/disagg_sectorprovs_2014.pkl"))
```

Then it creates a provincial `SimulationConfiguration`, builds the model from the
`DataWrapper`, runs it, and saves an H5 file:

```python
model = Simulation.from_datawrapper(
    datawrapper=data,
    simulation_configuration=config,
)

model.run()
model.save(
    save_dir=Path("path/to/outputs"),
    file_name="results_provinces_current_table.h5",
)
```

Source:

- [`macromodel/simulation.py`](../macromodel/simulation.py)
- [`macromodel/configurations/simulation_configuration.py`](../macromodel/configurations/simulation_configuration.py)

Notebook reference: see the runnable notebooks listed in [Section 0](#0-suggested-starting-point).

---

## 4. Where Raw Data Enters the Model

Raw data enters through `DataWrapper.from_config(...)`. The model run script only sees the
already-processed pickle.

### Step 1: `DataWrapper.from_config`

The relevant routing logic is in:

- [`macro_data/data_wrapper.py`](../macro_data/data_wrapper.py)

The Canada provincial case is activated when the configuration contains Canada as an
aggregated country with provinces:

```python
if configuration.aggregation_structure:
    regions_dict = configuration.aggregation_structure
    if Country("CAN") in configuration.countries and configuration.is_aggregated(Country("CAN")):
        use_provincial_can_reader = True
```

Then `DataReaders.from_raw_data(...)` is called with:

```python
use_disagg_can_2014_reader=configuration.can_disaggregation,
use_provincial_can_reader=use_provincial_can_reader,
regions_dict=regions_dict,
```

This is the point where the model switches from a national Canada run to the provincial
Canada reader path.

### Step 2: `DataReaders.from_raw_data`

The core reader orchestration is in:

- [`macro_data/readers/default_readers.py`](../macro_data/readers/default_readers.py)

This object loads:

- OECD ICIO tables
- the provincial Canada IO override
- WIOD SEA socioeconomic data
- exchange rates
- policy data
- emissions data
- household / population data
- finance and macroeconomic auxiliary data

The provincial IO table is loaded in the `use_provincial_can_reader` branch:

```python
disagg_path = raw_data_path / "icio" / "icio_2014_can_provinces.csv"
df = pd.read_csv(disagg_path, header=[0, 1], index_col=[0, 1])

df *= 1e6
```

For the current Canada provincial workflow, `icio_2014_can_provinces.csv` should be supplied
in **million USD**. The reader multiplies the table by `1e6` so the model receives full-unit
USD values.

---

## 5. Main Data Inputs and How They Are Used

### 5.1 Provincial IO table

Runtime path:

```text
path/to/raw_data/icio/icio_2014_can_provinces.csv
```

Loaded in:

- [`macro_data/readers/default_readers.py`](../macro_data/readers/default_readers.py)

Used to initialize:

- sectoral output
- intermediate input use
- value added
- taxes less subsidies
- household consumption by sector
- government consumption by sector
- fixed capital formation
- exports and imports
- interprovincial and external trade shares

The provincial reader replaces the 2014 ICIO reader’s internal IO matrix:

```python
icio[simulation_year].iot = df.sort_index()
icio[simulation_year].considered_countries = countries_and_regions
```

After this override, calls like:

```python
current_icio_reader.get_total_output(country_name)
current_icio_reader.get_value_added(country_name)
current_icio_reader.get_exports(country_name)
```

are reading from the provincial IO table.

### 5.2 WIOD SEA socioeconomic data

Runtime path:

```text
path/to/raw_data/wiod_sea/wiod_sea.csv
```

Loaded in:

- [`macro_data/readers/socioeconomic_data/wiod_sea_data.py`](../macro_data/readers/socioeconomic_data/wiod_sea_data.py)
- [`macro_data/readers/default_readers.py`](../macro_data/readers/default_readers.py)

SEA provides:

- value added
- labour compensation
- capital compensation
- capital stock

The SEA reader converts raw monetary values to USD-like full-unit values:

```python
stacked["Value"] /= stacked["country"].map(exchange_rates.exchange_rates_dict(year))
stacked["Value"] *= 1e6
```

In the current provincial model, SEA values are then bridged to the IO sector list and
province structure. The current bridge logic lives in:

- [`macro_data/readers/io_tables/sector_contracts.py`](../macro_data/readers/io_tables/sector_contracts.py)

### 5.3 Exchange rates

Runtime path:

```text
path/to/raw_data/exchange_rates/exchange_rates.csv
```

Loaded by:

- [`macro_data/readers/economic_data/exchange_rates.py`](../macro_data/readers/economic_data/exchange_rates.py)

Exchange rates are stored as:

```text
LCU per USD
```

For Canada in 2014, the important value used in the current discussion is approximately:

```text
1.104747 CAD per USD
```

### 5.4 Emissions and policy data

Important folders:

```text
path/to/raw_data/emission_factors/
path/to/raw_data/policy/
```

These feed:

- firm input emissions
- household consumption emissions
- household investment emissions
- consumer carbon-price schedules
- output-based pricing schedules
- sector exogenous price paths where configured

For public sharing, raw-data inventories should be distributed as separate documentation or
alongside the private raw-data bundle, not committed with the source repository.

### 5.5 Household and population data

The Canada beta workflow uses household and individual inputs to build synthetic households,
wealth, income, debt, deposits, and consumption behaviour. These data pass through HFCS and
synthetic population readers.

Relevant code areas:

- [`macro_data/readers/population_data/hfcs_reader.py`](../macro_data/readers/population_data/hfcs_reader.py)
- [`macro_data/processing/synthetic_population/`](../macro_data/processing/synthetic_population/)
- [`macro_data/processing/synthetic_matching/`](../macro_data/processing/synthetic_matching/)

---

## 6. Currency and Unit Contract

For the current Canada provincial workflow, the IO table input is expected in **million
USD**. During reading, the table is converted to full-unit USD. Downstream model data then
stores both USD and local-currency versions for the main monetary variables.

Relevant code:

- [`macro_data/readers/util/industry_extraction.py`](../macro_data/readers/util/industry_extraction.py)

```python
"Output in USD": current_icio_reader.get_total_output(country_name),
"Output in LCU": current_icio_reader.get_total_output(country_name) * exchange_rate,

"Value Added in USD": current_icio_reader.get_value_added(country_name),
"Value Added in LCU": exchange_rate * current_icio_reader.get_value_added(country_name),
```

For Canada, local-currency values are CAD. The exchange-rate reader stores exchange rates as
`LCU per USD`, so Canada LCU fields are obtained by multiplying USD fields by the CAD/USD
exchange rate. Output plots in the sample notebooks should be read as model diagnostics
unless the notebook explicitly labels the plotted series as USD or CAD.

---

## 7. How the Model Object Is Built

After readers load raw data, the model constructs synthetic countries. For the provincial
Canada case, each province becomes a `SyntheticCountry` before it becomes a runtime
`Country` inside the simulation.

Important path:

```text
Raw data
  -> DataReaders
  -> industry vectors and matrices
  -> SyntheticCountry objects
  -> DataWrapper pickle
  -> Simulation.from_datawrapper(...)
  -> runtime Country objects
```

The `DataWrapper` pickle does not simply store raw tables. It stores the initialized model
state needed to construct:

- firms
- households
- banks
- central bank
- governments
- goods market
- labour market
- housing market
- credit market
- trade proportions
- exogenous time series
- emissions factors

---

## 8. Key Model Components

### Firms

Firms are initialized by sector. In the current beta configuration, the script uses:

```python
base_config.single_firm_per_industry = True
base_config.firms_configuration.constructor = "Default"
```

Each province therefore has one representative firm per sector in the current simplified
provincial beta run.

Firms use:

- sectoral output
- intermediate inputs
- labour compensation
- capital input structures
- inventories
- deposits, debt, equity
- taxes
- productivity and technical-coefficient functions

Relevant code:

- [`macro_data/processing/synthetic_firms/`](../macro_data/processing/synthetic_firms/)
- [`macromodel/agents/firms/`](../macromodel/agents/firms/)

### Households and individuals

Households and individuals are initialized from household finance and population data.

They carry:

- income
- consumption
- deposits
- debt
- housing-related balance sheet variables
- consumption weights
- investment behaviour

Relevant code:

- [`macro_data/processing/synthetic_population/`](../macro_data/processing/synthetic_population/)
- [`macromodel/agents/households/`](../macromodel/agents/households/)
- [`macromodel/agents/individuals/`](../macromodel/agents/individuals/)

### Banks and credit

Banks are initialized as financial intermediaries and participate in the credit market.

The beta provincial script currently uses:

```python
base_config.single_bank = True
```

Relevant code:

- [`macro_data/processing/synthetic_banks/`](../macro_data/processing/synthetic_banks/)
- [`macromodel/agents/banks/`](../macromodel/agents/banks/)
- [`macromodel/markets/credit_market/`](../macromodel/markets/credit_market/)

### Governments

The current beta setup uses one government entity per province:

```python
base_config.single_government_entity = True
```

Government entities consume goods, collect/pay taxes, and can be used for policy experiments.

Relevant code:

- [`macro_data/processing/synthetic_government_entities/`](../macro_data/processing/synthetic_government_entities/)
- [`macromodel/agents/government_entities/`](../macromodel/agents/government_entities/)

### Markets

The runtime model includes:

- goods market
- labour market
- credit market
- housing market

Relevant code:

- [`macromodel/markets/goods_market/`](../macromodel/markets/goods_market/)
- [`macromodel/markets/labour_market/`](../macromodel/markets/labour_market/)
- [`macromodel/markets/credit_market/`](../macromodel/markets/credit_market/)
- [`macromodel/markets/housing_market/`](../macromodel/markets/housing_market/)

---

## 9. Running and Inspecting Outputs

### Simple beta run

Local run scripts commonly write an H5 file such as:

```text
path/to/outputs/results_provinces_current_table.h5
```

Notebook reference: see the runnable notebooks listed in [Section 0](#0-suggested-starting-point).

The output is an H5 file with one top-level group per province plus `ROW` and global market
groups. A typical top-level structure is:

```text
CAN_AB
CAN_BC
CAN_MB
CAN_NB
CAN_NL
CAN_NS
CAN_ON
CAN_PE
CAN_QC
CAN_SK
GM
ROW
```

Inside a province group, common components include:

```text
economy
firms
households
labour_market
housing_market
banks
central_government
government_entities
```

Useful economy-level fields include:

- `gdp_output`
- `gdp_income`
- `gdp_expenditure`
- `exports`
- `imports`
- `cpi`
- `firm_insolvency_rate`
- `household_insolvency_rate`

Useful firm-level fields include:

- `production`
- `demand`
- `gross_fixed_capital_formation`
- `profits`
- `labour_inputs`
- `capital_inputs_stock_value`

Useful household-level fields include:

- `consumption`
- `income`
- `investment`
- `debt`

### Example inspection snippet

```python
import h5py
import numpy as np

path = "path/to/outputs/results_provinces_current_table.h5"
provinces = [
    "CAN_AB", "CAN_BC", "CAN_MB", "CAN_NB", "CAN_NL",
    "CAN_NS", "CAN_ON", "CAN_PE", "CAN_QC", "CAN_SK",
]

with h5py.File(path, "r") as f:
    national_gdp = []
    for t in range(f["CAN_ON"]["economy"]["gdp_output"].shape[0]):
        total = sum(
            float(np.nansum(f[p]["economy"]["gdp_output"][t]))
            for p in provinces
        )
        national_gdp.append(total)

print(national_gdp)
```

## 10. Final Note On Testing And Policy Runs

MacroABM-CA results, like other similar models, are sensitive to model settings, input data, and policy assumptions. When using the model for beta testing or policy
simulation, document the raw data version, IO table, run script or notebook, timestep,
random seed or trial design, policy settings, and any code or parameter changes made for
the run. This makes results easier to interpret, compare, reproduce, and revise as the
model continues to develop.
