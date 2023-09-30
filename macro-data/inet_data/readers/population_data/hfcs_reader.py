import pandas as pd

from pathlib import Path
from inet_data.readers.economic_data.exchange_rates import WorldBankRatesReader


var_mapping = {
    "ID": "ID",
    "id": "ID",
    "HID": "Corresponding Household ID",
    "hid": "Corresponding Household ID",
    "iid": "Corresponding Individuals ID",
    "HW0010": "Weight",
    "DHHTYPE": "Type",
    "RA0200": "Gender",
    "RA0300": "Age",
    "PA0200": "Education",
    "PE0100a": "Labour Status",
    "PE0400": "Employment Industry",
    "PG0110": "Employee Income",
    "PG0210": "Self-Employment Income",
    "DI1300": "Rental Income from Real Estate",
    "DI1400": "Income from Financial Assets",
    "DI1500": "Income from Pensions",
    "DI1620": "Regular Social Transfers",
    "DI2000": "Income",
    "PG0510": "Income from Unemployment Benefits",
    "DA1110": "Value of the Main Residence",
    "DA1120": "Value of other Properties",
    "DA1130": "Value of Household Vehicles",
    "DA1131": "Value of Household Valuables",
    "DA1140": "Value of Self-Employment Businesses",
    "DA2101": "Wealth in Deposits",
    "DA2102": "Mutual Funds",
    "DA2103": "Bonds",
    "DA2104": "Value of Private Businesses",
    "DA2105": "Shares",
    "DA2106": "Managed Accounts",
    "DA2107": "Money owed to Households",
    "DA2108": "Other Assets",
    "DA2109": "Voluntary Pension",
    "DL1110": "Outstanding Balance of HMR Mortgages",
    "DL1120": "Outstanding Balance of Mortgages on other Properties",
    "DL1210": "Outstanding Balance of Credit Line",
    "DL1220": "Outstanding Balance of Credit Card Debt",
    "DL1230": "Outstanding Balance of other Non-Mortgage Loans",
    "HB0300": "Tenure Status of the Main Residence",
    "HB2300": "Rent Paid",
    "HB2410": "Number of Properties other than Household Main Residence",
    "DOCOGOODP": "Consumption of Consumer Goods/Services as a Share of Income",
    "HI0220": "Amount spent on Consumption of Goods and Services",
}

var_numerical = [
    "Income",
    "Employee Income",
    "Self-Employment Income",
    "Rental Income from Real Estate",
    "Income from Financial Assets",
    "Income from Pensions",
    "Regular Social Transfers",
    "Income from Unemployment Benefits",
    "Value of the Main Residence",
    "Value of other Properties",
    "Value of Household Vehicles",
    "Value of Household Valuables",
    "Value of Self-Employment Businesses",
    "Wealth in Deposits",
    "Mutual Funds",
    "Bonds",
    "Value of Private Businesses",
    "Shares",
    "Managed Accounts",
    "Money owed to Households",
    "Other Assets",
    "Voluntary Pension",
    "Outstanding Balance of HMR Mortgages",
    "Outstanding Balance of Mortgages on other Properties",
    "Outstanding Balance of Credit Line",
    "Outstanding Balance of Credit Card Debt",
    "Outstanding Balance of other Non-Mortgage Loans",
    "Rent Paid",
    "Amount spent on Consumption of Goods and Services",
]


class HFCSReader:
    def __init__(
        self,
        country_name_short: str,
        individuals_df: pd.DataFrame,
        households_df: pd.DataFrame,
    ):
        self.country_name_short = country_name_short
        self.individuals_df = individuals_df
        self.households_df = households_df

    @classmethod
    def from_csv(
        cls,
        country_name: str,
        country_name_short: str,
        year: int,
        hfcs_data_path: Path,
        exchange_rates: WorldBankRatesReader,
        num_surveys: int = 5,
    ) -> "HFCSReader":
        # Take default paths
        individuals_paths = [hfcs_data_path / str(year) / ("P" + str(i) + ".csv") for i in range(1, num_surveys + 1)]
        households_paths = [hfcs_data_path / str(year) / ("H" + str(i) + ".csv") for i in range(1, num_surveys + 1)]
        derived_paths = [hfcs_data_path / str(year) / ("D" + str(i) + ".csv") for i in range(1, num_surveys + 1)]

        # Read inet_data on individuals
        if len(individuals_paths) > 0:
            individuals_df = pd.concat(
                [
                    cls.read_csv(
                        path=ind_path,
                        country_name=country_name,
                        country_name_short=country_name_short,
                        year=year,
                        exchange_rates=exchange_rates,
                    )
                    for ind_path in individuals_paths
                ],
                axis=0,
            )
        else:
            individuals_df = pd.DataFrame()

        # Read inet_data on households
        if len(households_paths) > 0:
            households_df = pd.concat(
                [
                    cls.read_csv(
                        path=hh_path,
                        country_name=country_name,
                        country_name_short=country_name_short,
                        year=year,
                        exchange_rates=exchange_rates,
                    )
                    for hh_path in households_paths
                ],
                axis=0,
            )
        else:
            households_df = pd.DataFrame()

        # Read derived inet_data
        if len(derived_paths) > 0:
            derived_df = pd.concat(
                [
                    cls.read_csv(
                        path=der_path,
                        country_name=country_name,
                        country_name_short=country_name_short,
                        year=year,
                        exchange_rates=exchange_rates,
                    )
                    for der_path in derived_paths
                ],
                axis=0,
            )
            derived_df.drop("Weight", axis=1, inplace=True)
        else:
            derived_df = pd.DataFrame()

        # Join the derived inet_data with the household inet_data
        households_df = households_df.join(derived_df)

        return cls(
            country_name_short=country_name_short,
            individuals_df=individuals_df,
            households_df=households_df,
        )

    @staticmethod
    def read_csv(
        path: Path | str,
        country_name: str,
        country_name_short: str,
        year: int,
        exchange_rates: WorldBankRatesReader,
    ) -> pd.DataFrame:
        # Load inet_data
        df = pd.read_csv(path, encoding="unicode_escape", low_memory=False)

        # Cosmetics
        df = df[df["SA0100"] == country_name_short]
        df = df[[col for col in var_mapping.keys() if col in df.columns]]
        df.rename(columns=var_mapping, inplace=True)
        df.set_index("ID", inplace=True)
        df.replace("A", 0.0, inplace=True)

        # Change currencies to the local currency
        var_numerical_union = [v for v in var_numerical if v in df.columns]
        df.loc[:, var_numerical_union] = df.loc[:, var_numerical_union].astype(float)
        df.loc[:, var_numerical_union] = exchange_rates.from_eur_to_lcu(
            country=country_name,
            year=year,
        ) * df.loc[
            :, var_numerical_union
        ].replace("A", 0.0).fillna(0.0)
        return df
