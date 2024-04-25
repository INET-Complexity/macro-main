from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.readers.default_readers import DataReaders
from macro_data.readers.util.industry_extraction import compile_exogenous_industry_data


@dataclass
class ExogenousCountryData:
    country_name: Country
    inflation: pd.DataFrame
    national_accounts: pd.DataFrame
    labour_stats: pd.DataFrame
    house_price_index: pd.DataFrame

    @classmethod
    def from_data_readers(
        cls,
        readers: DataReaders,
        country_name: Country,
        industry_vectors: pd.DataFrame,
        year: int,
        quarter: int,
        proxy_country: Optional[Country] = None,
    ):
        inflation = readers.imf_reader.get_inflation(country_name)
        if inflation is None:
            inflation = readers.world_bank.get_inflation(country_name)

        national_accounts_growth = readers.get_national_accounts_growth(country_name)

        if proxy_country is None:
            capital_formation_tax = readers.eurostat.taxrate_on_capital_formation(country_name, year)
        else:
            capital_formation_tax = readers.eurostat.taxrate_on_capital_formation(proxy_country, year)

        national_accounts_data = compile_national_accounts_data(
            national_accounts_growth,
            industry_vectors,
            inflation,
            readers.world_bank.get_tau_vat(country_name, year),
            capital_formation_tax,
            readers.world_bank.get_tau_exp(country_name, year),
            year,
            quarter,
        )

        labour_stats = prepare_labour_stats(country_name, readers)

        house_price_index = readers.oecd_econ.get_house_price_index(country_name)

        return cls(
            country_name=country_name,
            inflation=inflation,
            national_accounts=national_accounts_data,
            labour_stats=labour_stats,
            house_price_index=house_price_index,
        )

    def get_calibration_data(self, year: int, quarter: int):
        index = self.national_accounts.index
        unemployment = self.labour_stats[["Unemployment Rate (Value)", "Unemployment Rate (Growth)"]].reindex(index)

        vacancies = self.labour_stats[["Vacancy Rate (Value)", "Vacancy Rate (Growth)"]].reindex(index)

        house_price_index = self.house_price_index.reindex(index)["Nominal House Price Index Growth"]
        house_price_index.name = "HPI (Growth)"
        house_price_index = pd.DataFrame(house_price_index)

        house_price_index["HPI (Value)"] = normalised_growth(house_price_index["HPI (Growth)"], year, quarter)

        country_data = pd.concat(
            (
                self.national_accounts,
                unemployment,
                vacancies,
                house_price_index,
            ),
            axis=1,
        )

        country_data.columns = pd.MultiIndex.from_product([[self.country_name], country_data.columns])

        return country_data


def prepare_labour_stats(country_name: Country, readers: DataReaders):
    labour_stats = readers.imf_reader.get_labour_stats(country_name)
    vacancy_rate = readers.oecd_econ.get_vacancy_rate(country_name)
    participation_rate = readers.world_bank.get_participation_rate(country_name)
    unemployment_rate = readers.world_bank.get_unemployment_rate(country_name)
    if labour_stats is None:
        labour_stats = pd.concat(
            (
                vacancy_rate.reindex(unemployment_rate.index),
                participation_rate.reindex(unemployment_rate.index),
                unemployment_rate,
            ),
            axis=1,
        )
    else:
        labour_stats = pd.concat(
            (
                labour_stats,
                vacancy_rate.reindex(labour_stats.index),
                participation_rate.reindex(labour_stats.index),
            ),
            axis=1,
        )
    labour_stats = labour_stats.ffill(axis=0)

    labour_stats.rename(
        columns={
            "Unemployment Rate": "Unemployment Rate (Value)",
            "Participation Rate": "Participation Rate (Value)",
            "Vacancy Rate": "Vacancy Rate (Value)",
        },
        inplace=True,
    )
    labour_stats["Unemployment Rate (Growth)"] = labour_stats["Unemployment Rate (Value)"].pct_change(fill_method=None)
    labour_stats["Participation Rate (Growth)"] = labour_stats["Participation Rate (Value)"].pct_change(
        fill_method=None
    )
    labour_stats["Vacancy Rate (Growth)"] = labour_stats["Vacancy Rate (Value)"].pct_change(fill_method=None)

    return labour_stats


def create_all_exogenous_data(
    readers: DataReaders,
    country_names: list[Country],
    year_min: int = 2010,
    year_max: int = 2019,
    proxy_countries: dict[Country, Country] = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Create exogenous data for each country in the given list of country names.

    This data includes:
        - log inflation
        - sectoral growth
        - unemployment rate
        - house price index
        - vacancy rate
        - total firm deposits and debt
        - industry data from the input-output tables

    Args:
        readers (DataReaders): An instance of the DataReaders class that provides access to various data sources.
        country_names (list[str]): A list of country names for which exogenous data needs to be created.
        year_min (int, optional): The minimum year for which exogenous data should be collected. Defaults to 2010.
        year_max (int, optional): The maximum year for which exogenous data should be collected. Defaults to 2019.
        proxy_countries (dict[str, str], optional): A dictionary of country names and their corresponding proxy EU
            countries. Defaults to None.

    Returns:
        dict[str, dict[str, pd.DataFrame]]: A dictionary containing exogenous data for each country, organized by country name and data type.
    """
    exogenous_industry_data = compile_exogenous_industry_data(readers, country_names, year_min, year_max)

    if proxy_countries is None:
        proxy_countries = {}

    for country in country_names:
        if country not in proxy_countries:
            proxy_countries[country] = country

    # get the set intersection of country_names and the keys of exogenous_industry_data
    exog_countries = list(set(country_names).intersection(exogenous_industry_data.keys()))
    # TODO this is a hack; sectoral growth needs to be readjusted
    exogenous_data = {
        country_name: {
            "inflation": prepare_inflation(country_name, readers),
            "log_inflation": readers.world_bank.get_log_inflation(country_name),
            "sectoral_growth": readers.eurostat.get_perc_sectoral_growth(proxy_countries[country_name]),
            "unemployment_rate": readers.oecd_econ.get_unemployment_rate(country_name),
            "house_price_index": readers.oecd_econ.get_house_price_index(country_name),
            "vacancy_rate": readers.oecd_econ.get_vacancy_rate(country_name),
            "total_firm_deposits_and_debt": readers.eurostat.get_total_industry_debt_and_deposits(
                country_name, proxy_countries[country_name]
            ),
            "iot_industry_data": exogenous_industry_data[country_name],
        }
        for country_name in exog_countries
    }

    return exogenous_data


def prepare_inflation(country_name: Country, readers: DataReaders):
    inflation = readers.imf_reader.get_inflation(country_name)
    if inflation is None:
        inflation = readers.world_bank.get_log_inflation(country_name)

    inflation[inflation == 0.0] = np.nan
    inflation.loc[np.isnan(inflation["PPI Inflation"]), "PPI Inflation"] = inflation.loc[
        np.isnan(inflation["PPI Inflation"]), "CPI Inflation"
    ]
    inflation.loc[np.isnan(inflation["CPI Inflation"]), "CPI Inflation"] = inflation.loc[
        np.isnan(inflation["CPI Inflation"]), "PPI Inflation"
    ]
    return inflation


def normalised_growth(growth_rates: pd.Series, year: int, quarter: int):
    growth = (1 + growth_rates).cumprod()
    return (growth / growth.loc[f"{year}-Q{quarter}"].values).values


def compile_national_accounts_data(
    national_accounts_growth: pd.DataFrame,
    industry_vectors: pd.DataFrame,
    inflation: pd.DataFrame,
    initial_vat: float,
    initial_cf_tax: float,
    initial_exports_tax_paid: float,
    base_year: int,
    base_quarter: int,
):
    inflation = inflation.reindex(national_accounts_growth.index).fillna(0.0)
    initial_taxes_on_products = (
        industry_vectors["Taxes Less Subsidies in LCU"].sum()
        + initial_vat * industry_vectors["Household Consumption in LCU"].sum()
        + initial_cf_tax * industry_vectors["Household Capital Inputs in LCU"].sum()
        + initial_exports_tax_paid
    )

    gross_fixed_capital_formation = (1 + initial_cf_tax) * industry_vectors[
        "Household Capital Inputs in LCU"
    ].sum() + industry_vectors["Firm Capital Inputs in LCU"].sum()

    gross_operating_surplus = (
        industry_vectors["Output in LCU"].sum()
        - industry_vectors["Labour Compensation in LCU"].sum()
        - industry_vectors["Intermediate Inputs Use in LCU"].sum()
        - industry_vectors["Taxes Less Subsidies in LCU"].sum()
    )

    # An initial sanity check
    gdp_output = (
        industry_vectors["Output in LCU"].sum()
        - industry_vectors["Taxes Less Subsidies in LCU"].sum()
        - industry_vectors["Intermediate Inputs Use in LCU"].sum()
        + initial_taxes_on_products
    )
    gdp_expenditure = (
        (1 + initial_vat) * industry_vectors["Household Consumption in LCU"].sum()
        + industry_vectors["Government Consumption in LCU"].sum()
        + (1 + initial_cf_tax) * industry_vectors["Household Capital Inputs in LCU"].sum()
        + industry_vectors["Firm Capital Inputs in LCU"].sum()
        + initial_exports_tax_paid
        + industry_vectors["Exports in LCU"].sum()
        - industry_vectors["Imports in LCU"].sum()
    )
    gdp_income = (
        initial_taxes_on_products + gross_operating_surplus + industry_vectors["Labour Compensation in LCU"].sum()
    )

    def get_growth(column: str):
        if column in national_accounts_growth.columns:
            return normalised_growth(national_accounts_growth[column], base_year, base_quarter)
        else:
            return normalised_growth(inflation[column], base_year, base_quarter)

    assert np.isclose(gdp_output, gdp_expenditure)
    assert np.isclose(gdp_output, gdp_income)

    national_accounts_data = {
        "PPI (Growth)": inflation["PPI Inflation"].values,
        "PPI (Value)": get_growth("PPI Inflation"),
        "CPI (Growth)": inflation["CPI Inflation"].values,
        "CPI (Value)": get_growth("CPI Inflation"),
        #
        "GDP (Growth)": national_accounts_growth["GDP"].values,
        "GDP (Value)": get_growth("GDP")
        * (
            industry_vectors["Output in LCU"].sum()
            - industry_vectors["Taxes Less Subsidies in LCU"].sum()
            - industry_vectors["Intermediate Inputs Use in LCU"].sum()
            + initial_taxes_on_products
        ),
        "Real GDP (Growth)": (1 + national_accounts_growth["GDP"].values) / (1 + inflation["PPI Inflation"].values)
        - 1.0,
        "Real GDP (Value)": get_growth("GDP")
        / get_growth("PPI Inflation")
        * (
            industry_vectors["Output in LCU"].sum()
            - industry_vectors["Intermediate Inputs Use in LCU"].sum()
            + initial_taxes_on_products
        ),
        #
        "Gross Output (Growth)": national_accounts_growth["Gross Output"].values,
        "Gross Output (Value)": get_growth("Gross Output") * industry_vectors["Output in LCU"].sum(),
        "Real Gross Output (Growth)": (1 + national_accounts_growth["Gross Output"].values)
        / (1 + inflation["PPI Inflation"].values)
        - 1.0,
        "Real Gross Output (Value)": get_growth("Gross Output")
        / get_growth("PPI Inflation")
        * industry_vectors["Output in LCU"].sum(),
        "Intermediate Consumption (Growth)": national_accounts_growth["Intermediate Consumption"].values,
        "Intermediate Consumption (Value)": get_growth("Intermediate Consumption")
        * industry_vectors["Intermediate Inputs Use in LCU"].sum(),
        "Taxes less Subsidies on Products (Growth)": national_accounts_growth[
            "Taxes less Subsidies on Production"
        ].values,
        "Taxes less Subsidies on Products (Value)": get_growth("Taxes less Subsidies on Production")
        * initial_taxes_on_products,
        "Taxes less Subsidies on Production (Growth)": national_accounts_growth[
            "Taxes less Subsidies on Production"
        ].values,
        "Taxes less Subsidies on Production (Value)": get_growth("Taxes less Subsidies on Production")
        * industry_vectors["Taxes Less Subsidies in LCU"].sum(),
        #
        "Household Consumption (Growth)": national_accounts_growth["HH Cons"].values,
        "Household Consumption (Value)": get_growth("HH Cons")
        * (1 + initial_vat)
        * industry_vectors["Household Consumption in LCU"].sum(),
        "Real Household Consumption (Growth)": (1 + national_accounts_growth["HH Cons"].values)
        / (1 + inflation["PPI Inflation"].values)
        - 1.0,
        "Real Household Consumption (Value)": get_growth("HH Cons")
        / get_growth("PPI Inflation")
        * (1 + initial_vat)
        * industry_vectors["Household Consumption in LCU"].sum(),
        #
        "Household Investment (Growth)": national_accounts_growth["HH Cons"].values,
        "Household Investment (Value)": get_growth("HH Cons")
        * (1 + initial_cf_tax)
        * industry_vectors["Household Capital Inputs in LCU"].sum(),
        "Real Household Investment (Growth)": (1 + national_accounts_growth["HH Cons"].values)
        / (1 + inflation["PPI Inflation"].values)
        - 1.0,
        "Real Household Investment (Value)": get_growth("HH Cons")
        / get_growth("PPI Inflation")
        * (1 + initial_cf_tax)
        * industry_vectors["Household Capital Inputs in LCU"].sum(),
        #
        "Government Consumption (Growth)": national_accounts_growth["Gov Cons"].values,
        "Government Consumption (Value)": get_growth("Gov Cons")
        * industry_vectors["Government Consumption in LCU"].sum(),
        "Real Government Consumption (Growth)": (1 + national_accounts_growth["Gov Cons"].values)
        / (1 + inflation["PPI Inflation"].values)
        - 1.0,
        "Real Government Consumption (Value)": get_growth("Gov Cons")
        / get_growth("PPI Inflation")
        * industry_vectors["Government Consumption in LCU"].sum(),
        "Gross Fixed Capital Formation (Growth)": national_accounts_growth["Gross Fixed Capital Formation"].values,
        "Gross Fixed Capital Formation (Value)": get_growth("Gross Fixed Capital Formation")
        * gross_fixed_capital_formation,
        "Changes in Inventories (Growth)": np.zeros_like(national_accounts_growth["GDP"].values),
        "Changes in Inventories (Value)": np.zeros_like(national_accounts_growth["GDP"].values),
        "Exports (Growth)": national_accounts_growth["Exports"].values,
        "Exports (Value)": get_growth("Exports")
        * (industry_vectors["Exports in LCU"].sum() + initial_exports_tax_paid),
        "Exports before Taxes (Value)": get_growth("Exports") * industry_vectors["Exports in LCU"].sum(),
        "Imports (Growth)": national_accounts_growth["Imports"].values,
        "Imports (Value)": get_growth("Imports") * industry_vectors["Imports in LCU"].sum(),
        #
        "Compensation of Employees (Growth)": national_accounts_growth["Compensation of Employees"].values,
        "Compensation of Employees (Value)": get_growth("Compensation of Employees")
        * industry_vectors["Labour Compensation in LCU"].sum(),
        "Gross Operating Surplus and Mixed Income (Growth)": national_accounts_growth[
            "Gross Operating Surplus and Mixed Income"
        ].values,
        "Gross Operating Surplus and Mixed Income (Value)": get_growth("Gross Operating Surplus and Mixed Income")
        * gross_operating_surplus,
        "Gross Value Added (Growth)": national_accounts_growth["Gross Value Added"].values,
        "Gross Value Added (Value)": get_growth("Gross Value Added") * industry_vectors["Value Added in LCU"].sum(),
        "Gross Value Added - A (Growth)": national_accounts_growth["Gross Value Added - A"].values,
        "Gross Value Added - A (Value)": get_growth("Gross Value Added - A")
        * industry_vectors["Value Added in LCU"].iloc[[0]].sum(),
        "Gross Value Added - B, C, D, E (Growth)": national_accounts_growth["Gross Value Added - B, C, D, E"].values,
        "Gross Value Added - B, C, D, E (Value)": get_growth("Gross Value Added - B, C, D, E")
        * industry_vectors["Value Added in LCU"].iloc[[1, 2, 3, 4]].sum(),
        "Gross Value Added - C (Growth)": national_accounts_growth["Gross Value Added - C"].values,
        "Gross Value Added - C (Value)": get_growth("Gross Value Added - C")
        * industry_vectors["Value Added in LCU"].iloc[[2]].sum(),
        "Gross Value Added - F (Growth)": national_accounts_growth["Gross Value Added - F"].values,
        "Gross Value Added - F (Value)": get_growth("Gross Value Added - F")
        * industry_vectors["Value Added in LCU"].iloc[[5]].sum(),
        "Gross Value Added - G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U (Growth)": national_accounts_growth[
            "Gross Value Added - G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U"
        ].values,
        "Gross Value Added - G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U (Value)": get_growth(
            "Gross Value Added - G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U"
        )
        * industry_vectors["Value Added in LCU"].iloc[[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]].sum(),
        "Gross Value Added - G, H, I (Growth)": national_accounts_growth["Gross Value Added - G, H, I"].values,
        "Gross Value Added - G, H, I (Value)": get_growth("Gross Value Added - G, H, I")
        * industry_vectors["Value Added in LCU"].iloc[[6, 7, 8]].sum(),
        "Gross Value Added - J (Growth)": national_accounts_growth["Gross Value Added - J"].values,
        "Gross Value Added - J (Value)": get_growth("Gross Value Added - J")
        * industry_vectors["Value Added in LCU"].iloc[[9]].sum(),
        "Gross Value Added - K (Growth)": national_accounts_growth["Gross Value Added - K"].values,
        "Gross Value Added - K (Value)": get_growth("Gross Value Added - K")
        * industry_vectors["Value Added in LCU"].iloc[[10]].sum(),
        "Gross Value Added - L (Growth)": national_accounts_growth["Gross Value Added - L"].values,
        "Gross Value Added - L (Value)": get_growth("Gross Value Added - L")
        * industry_vectors["Value Added in LCU"].iloc[[11]].sum(),
        "Gross Value Added - M, N (Growth)": national_accounts_growth["Gross Value Added - M, N"].values,
        "Gross Value Added - M, N (Value)": get_growth("Gross Value Added - M, N")
        * industry_vectors["Value Added in LCU"].iloc[[12, 13]].sum(),
        "Gross Value Added - O, P, Q (Growth)": national_accounts_growth["Gross Value Added - O, P, Q"].values,
        "Gross Value Added - O, P, Q (Value)": get_growth("Gross Value Added - O, P, Q")
        * industry_vectors["Value Added in LCU"].iloc[[14, 15, 16]].sum(),
        "Gross Value Added - R, S, T, U (Growth)": national_accounts_growth["Gross Value Added - R, S, T, U"].values,
        "Gross Value Added - R, S, T, U (Value)": get_growth("Gross Value Added - R, S, T, U")
        * industry_vectors["Value Added in LCU"].iloc[[17]].sum(),
    }

    return pd.DataFrame(data=national_accounts_data, index=national_accounts_growth.index)
