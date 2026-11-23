# ECBReader

`ECBReader` loads European Central Bank (ECB) lending-rate data for firms and households. It reads three CSV files, converts country columns from two-letter to three-letter ISO codes, resamples observations to quarterly frequency, and returns interest rates as decimals.

## Expected files

Place the following CSV files in the directory passed to `ECBReader`:

- `firm_loans.csv` - Monthly interest rates on loans in euros to non-financial corporations (new business), measured in percent per annum,

API: https://data-api.ecb.europa.eu/service/data/MIR

SERIES_SUFFIX = "B.A2A.A.R.A.2240.EUR.N"

- `household_loans_for_consumption.csv` - Household-consumption loan series

API = https://data-api.ecb.europa.eu/service/data/MIR

SERIES_SUFFIX = "B.A2B.A.R.A.2250.EUR.N"

- `household_loans_for_mortgages.csv` - Mortgage interest-rate data (A2C)

API = "https://data-api.ecb.europa.eu/service/data/MIR"

SERIES_SUFFIX = "B.A2C.A.R.A.2250.EUR.N"

Each CSV is expected to contain a `DATE` column and ECB country-rate columns. The Euro area aggregate column (`U2`) is removed during preprocessing.

## Constructor

```python
from pathlib import Path
from macro_data.configuration.countries import Country
from macro_data.readers.economic_data.ecb_reader import ECBReader

reader = ECBReader(
    path=Path("path/to/ecb/data"),
    proxy_country=Country("DEU"),
)
```

`proxy_country` stores the country used as a proxy, although the current implementation does not automatically apply it in the getter methods.

## Retrieving rates

Country names must use three-letter ISO codes.

```python
firm_rates = reader.get_firm_rates("FRA")
consumption_rates = reader.get_household_consumption_rates("FRA")
mortgage_rates = reader.get_household_mortgage_rates("FRA")
```

Each method returns a `pandas.Series` indexed by quarter, or `None` if the country is absent. Values are converted from percentages to decimals: for example, `5.0` becomes `0.05`.

## Preprocessing

`preprocess_df()` removes the `TIME PERIOD` column, extracts two-letter country codes from ECB column names, converts them to three-letter codes, removes `U2`, and resamples the data to the start-of-quarter (`QS`) frequency using mean aggregation.

## API reference

::: macro_data.readers.economic_data.ecb_reader
    options:
        members:
            - ECBReader
            - country_code_switch
            - preprocess_df
