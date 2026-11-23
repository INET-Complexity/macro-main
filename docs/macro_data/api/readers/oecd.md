# OECDEconData

The `OECDEconData` reader loads OECD macroeconomic and labour-market series used to calibrate and supplement the model. It is a broad reader for country-level indicators such as unemployment, vacancy rates, inflation, house prices, taxes, interest rates, and national accounts growth.

## Purpose

This reader is used to supply OECD-backed annual or quarterly time series for:

- unemployment rates
- vacancy rates
- inflation
- real and nominal house-price growth
- national accounts growth
- government debt and benefits ratios
- short- and long-term interest rates
- tax and bank-sector indicators

It is one of the main external data sources for exogenous macroeconomic calibration alongside IMF, World Bank, Eurostat, and ECB data.

## Data loaded

`OECDEconData.__init__()` loads a set of OECD CSV files from a directory. The mapping is defined in `get_files_with_codes()`, including datasets such as:

Expected files:
`ALFS_EMP.csv` Annual Labour Force Statistics (Employment by industry)

  API = https://sdmx.oecd.org/public/rest/data

  DATAFLOW = "OECD.SDD.TPS,DSD_ALFS@DF_ALFS_EMP_ISIC,1.1"

`BPF1.csv` Bank Profitability: Income Statement and Balance Sheet

Discontinued.

`consumption_by_income_quintiles.csv` Household consumption by income quintile.

Unit not available on OECD

`CTS_CIT.csv` Corporate income tax rates

  API: https://sdmx.oecd.org/public/rest/data/OECD.CTP.TPS,DSD_TAX_CIT@DF_CIT,/

*   CIT_RATE: central government rate

*   CIT_RATE_EXCL_SURTAX: central government rate without the surtax

*   CIT_RATE_LESS_SUB_NAT: central government rate after deducting sub-national taxes

*   SUB_CENT_GOV_CIT_RATE: state and local rate

*   COMB_CIT_RATE: central and sub-central combined

`DP_LIVE_21042023152525378.csv` Long-term interest rates

  API: https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_FINMARK,/

`DP_LIVE_21042023152700149.csv` Short-term interest rates

  API: https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_FINMARK,/

`DP_LIVE_PPI_INFL.csv` - Producer Price Index inflation

Discontinued. The new OECD data have no domestic-market PPI.

`DP_LIVE_UNEMP.csv` - Total unemployment benefits as a percentage of GDP

API: https://sdmx.oecd.org/public/rest/data/OECD.ELS.SPD,DSD_SOCX_AGG@DF_SOCX_AGG,/

`EGDNA_PUBLIC.csv` - Experimental Consumption Statistics

API: https://sdmx.oecd.org/public/rest". AGENCY = "OECD.SDD.NAD

`GINI_COEF.csv` - Gini coefficient

API: https://sdmx.oecd.org/public/rest. AGENCY, FLOW = "OECD.WISE.INE", "DSD_WISE_IDD@DF_IDD"

`HOUSE_PRICES.csv` - House price index

API: https://sdmx.oecd.org/public/rest/data/OECD.ECO.MPD,DSD_AN_HOUSE_PRICES@DF_HOUSE_PRICES,/all?dimensionAtObservation=AllDimensions&format=csvfile

`KEY.csv` - Key economic indicators

API: https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_KEI@DF_KEI,/all?dimensionAtObservation=AllDimensions&format=csvfile

`LAB_REG_VAC.csv` - Infra-annual registered unemployment and job vacancies

API1: https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_OLAB@DF_OIALAB_INDIC,/all?dimensionAtObservation=AllDimensions&format=csvfile

API2: https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_KEI@DF_KEI,/all?dimensionAtObservation=AllDimensions&format=csvfile

`bank_capital_requirements.csv` - Bank capital requirements

[Discontinued by OECD](https://www.oecd.org/en/publications/serials/bank-profitability-financial-statements-of-banks_g1gha515.html). 

`oecd_bank_capital_reqs.csv` - OECD bank data

[Discontinued by OECD](https://www.oecd.org/en/publications/serials/bank-profitability-financial-statements-of-banks_g1gha515.html).

`oecd_govt_debt_usd_ppp.csv` - Government debt in USD PPP

API1: https://sdmx.oecd.org/public/rest/data/OECD.GOV.GIP,DSD_GOV_FIN_INSTR@DF_GOV_FIN_INSTR_YU,/A.PRT+GBR+SWE+KOR+DNK+POL+HUN+BEL+COL+LUX+BRA+SVN+CAN+EST+ISR+SVK+CHE+NZL+CZE+DEU+ISL+CHL+LTU+ITA+ESP+AUS+HRV+JPN+AUT+IRL+ROU+ZAF+FIN+LVA+USA+NLD+MEX+FRA+BGR+NOR+TUR+GRC.GGD.USD_PPP.S13.F4+F2+F3+F1_F6_F8+_T.L.YU.PF?dimensionAtObservation=AllDimensions&format=csvfilewithlabels

API2: https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_KEI@DF_KEI,/all?dimensionAtObservation=AllDimensions&format=csvfile


- `QNA` - Quarterly National Accounts, national currency, current prices, quarterly levels, seasonally adjusted (old MEASURE = CQRSA)

API: https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_KEI@DF_KEI,/all?dimensionAtObservation=AllDimensions&format=csvfile

Discontinued codes: B2N_B3N (Net operating surplus and mixed income), K1 (Consumption of fixed capital)

`SDBS_BDI_ISIC4.csv` - Business Demography Indicators

API: https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_SDBSBD_ISIC4@

`SDBS_BDI_ISIC4_BIRTH.csv` -  Enterprise Births dataset

API: https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_KEI@DF_KEI,/all?dimensionAtObservation=AllDimensions&format=csvfile

`SDBS_BDI_ISIC4_DEATH.csv` -  Enterprise Deaths dataset

API: https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_KEI@DF_KEI,/all?dimensionAtObservation=AllDimensions&format=csvfile

`SNA_TABLE750_24042023172303744.csv` - General Government Debt 

API: https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_KEI@DF_KEI,/all?dimensionAtObservation=AllDimensions&format=csvfile

`SSIS_BSC_ISIC4.csv` - Business Sector by Industry (ISIC Rev. 4)

API: https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_KEI@DF_KEI,/all?dimensionAtObservation=AllDimensions&format=csvfile

`SOCX_AGG.csv` - Total social benefits as a percentage of GDP

API:https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_KEI@DF_KEI,/all?dimensionAtObservation=AllDimensions&format=csvfile

`STLABOUR.csv` - Size of active population

API: https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_KEI@DF_KEI,/all?dimensionAtObservation=AllDimensions&format=csvfile

`TABLE_I6.csv` -  Average personal income tax by family type

API: https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_KEI@DF_KEI,/all?dimensionAtObservation=AllDimensions&format=csvfile

DATASET: DSD_TAX_PIT@DF_PIT_AV_COMB

`TABLE_I7.csv` - Top statutory personal income tax rates by family type

DATASET: DSD_TAX_PIT@DF_PIT_TOP_EARN_THRESH


`TABLE_III1.csv` -  Employee contributions to social insurance by family type

DATASET: DSD_TAX_SSC@DF_SSC_EMPLOYEE

`TABLE_III2.csv` -  Employer contributions to social insurance by family type

DATASET: DSD_TAX_SSC@DF_SSC_EMPLOYER

`unemployment_rates.csv` - monthly unemployment-rate dataset 

API: https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M,/AUS+AUT+BEL+CAN+CZE+DNK+FIN+FRA+DEU+GRC+HUN+ISL+IRL+ITA+JPN+KOR+LUX+MEX+NLD+NZL+NOR+POL+PRT+SVK+ESP+SWE+CHE+TUR+GBR+USA+CHL+EST+ISR+SVN+OECD+G7+EA+LVA+LTU+COL+EU+CRI.UNE_LF_M.PT_LF_SUB._Z.N+Y._T+M+F.Y_GE15._Z.A+Q+M?dimensionAtObservation=AllDimensions&format=csvfile

DATASET: DSD_LFS@DF_IALFS_UNE_M

FOR TEST ONLY:
* `DP_LIVE_HH_SAV_RATE.csv`
* `DP_LIVE_HH_DISP_INC.csv`
* `HISTPOP.csv`
* `MEI_FIN.csv`
* `oecd_bank_data.csv`


## Common outputs

### Unemployment and vacancy rates

`get_unemployment_rate(country)` produces a quarterly DataFrame indexed by quarter with a single column:

- `Unemployment Rate`

It returns a decimal rate, not a percentage, so 0.05 means 5%.

`get_vacancy_rate(country)` similarly creates a quarterly vacancy-rate series, computed as vacancies divided by labour-force or active-population size.

### Inflation

`get_inflation(country)` reads OECD consumer and producer price data and returns a quarterly DataFrame with:

- `CPI Inflation`
- `PPI Inflation`

The values are period-over-period changes expressed as decimals.

### House-price growth

`get_house_price_index(country)` constructs real and nominal quarterly house-price growth series using OECD house-price indices. Output columns include:

- `Real House Price Index Growth`
- `Nominal House Price Index Growth`

### National accounts growth

`get_na_growth_rates(country)` reads quarterly national accounts from the OECD `QNA` dataset and converts the relevant variables into growth rates. It includes GDP and major expenditure categories, returning a quarterly DataFrame with columns such as:

- `GDP`
- `HH Cons`
- `Gov Cons`
- `Gross Fixed Capital Formation`
- `Exports`
- `Imports`
- `Gross Output`
- `Intermediate Consumption`

### Government and welfare indicators

This reader also exposes:

- `unemployment_benefits_gdp_pct(country, year)`
- `all_benefits_gdp_pct(country, year)`
- `general_gov_debt(country, year)`

These are used for welfare and public-finance calibration.

### Interest rates

`read_short_term_interest_rates(country, year)` and `read_long_term_interest_rates(country, year)` return annualised rates as decimals.

## Example usage

```python
from pathlib import Path
from macro_data.readers.economic_data.oecd_economic_data import OECDEconData
from macro_data.configuration.countries import Country

reader = OECDEconData(
    path=Path("path/to/oecd_data"),
    scale_dict={Country.FRANCE: 1000, Country.GERMANY: 1000},
)

unemployment = reader.get_unemployment_rate("FRA")
vacancies = reader.get_vacancy_rate(Country.FRANCE)
house_prices = reader.get_house_price_index("FRA")
inflation = reader.get_inflation("FRA")
na_growth = reader.get_na_growth_rates("FRA")

short_rate = reader.read_short_term_interest_rates(Country.FRANCE, 2020)
long_rate = reader.read_long_term_interest_rates(Country.FRANCE, 2020)
```

## Data conventions

- Most rate series are returned in decimal form.
- Time series are aligned to quarterly timestamps and indexed by `DatetimeIndex`.
- Wherever OECD data is incomplete, the reader uses fallback rules or fixed values for some countries.
- `prune()` can be used to filter data after a given date.

## API reference

::: macro_data.readers.economic_data.oecd_economic_data
    options:
        members:
            - OECDEconData
            - get_files_with_codes
            - get_household_consumption_by_income_quantile
            - get_unemployment_rate
            - get_vacancy_rate
            - get_house_price_index
            - get_inflation
            - get_na_growth_rates
            - unemployment_benefits_gdp_pct
            - all_benefits_gdp_pct
            - general_gov_debt
            - read_short_term_interest_rates
            - read_long_term_interest_rates
            - prune
