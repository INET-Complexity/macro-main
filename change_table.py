# %%
import pandas as pd

# %%
PATH = "/tests/test_macro_data/unit/sample_raw_data/icio/icio_can_2014_disagg_test.csv"

PATH_OLD = "/tests/test_macro_data/unit/sample_raw_data/icio/icio_can_2014_disagg.csv"

# %%
df = pd.read_csv(PATH_OLD, header=[0, 1], index_col=[0, 1])

# %%

cols_to_sum = [
    "Firm Fixed Capital Formation",
    "Household Fixed Capital Formation",
    "Government Consumption",
]

government_fraction = 0.16964695093993584


rest_fraction = 1 - government_fraction


# %%

df.loc[:, ("CAN", "Fixed Capital Formation")] = (
    df.loc[:, ("CAN", "Firm Fixed Capital Formation")] + df.loc[:, ("CAN", "Household Fixed Capital Formation")]
) / rest_fraction

df.loc[:, ("CAN", "Government Consumption")] -= df.loc[:, ("CAN", "Fixed Capital Formation")] * government_fraction

df.loc[df[("CAN", "Government Consumption")] < 0, ("CAN", "Government Consumption")] = 0

# %%

df.drop(columns=["Firm Fixed Capital Formation", "Household Fixed Capital Formation"], inplace=True)


# %%
df.to_csv(PATH)
# %%
