# %%
import numpy as np
import pandas as pd

ICIO_PATH = "/Users/jmoran/Projects/macrocosm/macromodel/macro-main/tests/test_macro_data/unit/sample_raw_data/icio/icio_2014_can_provinces.csv"


df = pd.read_csv(ICIO_PATH, index_col=[0, 1], header=[0, 1])

# %%

countries = df.columns.get_level_values(0).unique()
countries = list(set(countries) - {"ROW", "OUT"})

countries_with_row = countries + ["ROW"]


# %%


def pure_io(table: pd.DataFrame, countries: list[str]):
    return table.loc[countries, countries]


def get_table_output(table: pd.DataFrame, countries: list[str]):
    return pure_io(table, countries).sum(axis=0)


# %%
energy_table = "/Users/jmoran/Projects/macrocosm/macromodel/macro-main/tests/test_macro_data/unit/sample_raw_data/icio/icio_can_2014_disagg.csv"

energy_df = pd.read_csv(energy_table, index_col=[0, 1], header=[0, 1])

# %%
reduced_countries = ["CAN", "ROW"]

# %%
