from typing import Any

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer  # noqa
from sklearn.linear_model import LinearRegression

from inet_data.processing.synthetic_population.hfcs_household_tools import (
    set_household_types,
    set_household_housing_data,
)
from inet_data.processing.synthetic_population.hfcs_individual_tools import process_individual_data
from inet_data.processing.synthetic_population.synthetic_population import (
    SyntheticPopulation,
)
from inet_data.readers.default_readers import DataReaders
from inet_data.util.clean_data import remove_outliers
from inet_data.util.regressions import fit_linear


class SyntheticHFCSPopulation(SyntheticPopulation):
    def __init__(
        self,
        country_name: str,
        country_name_short: str,
        scale: int,
        year: int,
        industries: list[str],
        individual_data: pd.DataFrame,
        household_data: pd.DataFrame,
        social_housing_rent: float,
        coefficient_fa_income: float,
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
    ):
        saving_rates_model = LinearRegression()
        social_transfers_model = LinearRegression()
        wealth_distribution_model = LinearRegression()
        super().__init__(
            country_name,
            country_name_short,
            scale,
            year,
            industries,
            individual_data,
            household_data,
            social_housing_rent,
            coefficient_fa_income,
            consumption_weights,
            consumption_weights_by_income,
            saving_rates_model,
            social_transfers_model,
            wealth_distribution_model,
        )

    @classmethod
    def create_from_readers(
        cls,
        readers: DataReaders,
        country_name: str,
        country_name_short: str,
        scale: int,
        year: int,
        industry_data: dict[str, dict],
        industries: list[str],
        total_unemployment_benefits: float,
        rent_as_fraction_of_unemployment_rate: float,
        n_quantiles: int = 5,
    ):
        n_households = int(readers.eurostat.number_of_households(country_name, year) / scale)
        hfcs_individuals_data = readers.hfcs[country_name].individuals_df
        hfcs_households_data = readers.hfcs[country_name].households_df

        household_data, individual_data = sample_households(hfcs_households_data, hfcs_individuals_data, n_households)

        individual_data = process_individual_data(
            country_name, individual_data, industries, readers, scale, total_unemployment_benefits, year
        )
        n_unemployed = np.sum(individual_data["Activity Status"] == 2)

        household_data = remove_outliers(
            data=household_data,
            cols=[
                "Rent Paid",
                "Income",
                "Consumption of Consumer Goods/Services as a Share of Income",
            ],
        )
        household_data = household_data.loc[household_data["Corresponding Individuals ID"].notna()]

        household_data = set_household_types(household_data, individual_data)

        household_data = set_household_housing_data(
            household_data,
            scale,
            rent_as_fraction_of_unemployment_rate,
            unemployment_benefits_by_capita=total_unemployment_benefits / n_unemployed,
        )

        # initialise fields to nans, will be filled later when computing wealth
        non_initialised_fields = [
            "Wealth Other Real Assets",
            "Wealth in Real Assets",
            "Wealth in Other Financial Assets",
            "Wealth in Financial Assets",
            "Wealth",
            "Debt",
            "Net Wealth",
            "Employee Income",
        ]
        for field in non_initialised_fields:
            household_data[field] = np.nan

        social_housing_rent = rent_as_fraction_of_unemployment_rate * total_unemployment_benefits / n_unemployed
        consumption_weights = industry_data[country_name]["industry_vectors"]["Household Consumption Weights"].values
        consumption_weights_by_income = np.zeros((n_quantiles, len(consumption_weights)))
        for i in range(n_quantiles):
            consumption_weights_by_income[i] = consumption_weights

        return cls(
            country_name=country_name,
            country_name_short=country_name_short,
            scale=scale,
            year=year,
            industries=industries,
            household_data=household_data,
            individual_data=individual_data,
            social_housing_rent=social_housing_rent,
            consumption_weights=consumption_weights,
            consumption_weights_by_income=consumption_weights_by_income,
            coefficient_fa_income=0.0,
        )

    def restrict(self) -> None:
        self.household_data = self.household_data[
            [
                "Type",
                "Corresponding Individuals ID",
                "Corresponding Bank ID",
                "Corresponding Inhabited House ID",
                "Corresponding Renters",
                "Corresponding Property Owner",
                "Corresponding Additionally Owned Houses ID",
                #
                "Income",
                "Employee Income",
                "Regular Social Transfers",
                "Rental Income from Real Estate",
                "Income from Financial Assets",
                #
                "Saving Rate",
                #
                "Rent Paid",
                "Rent Imputed",
                #
                "Wealth",
                "Net Wealth",
                "Wealth in Real Assets",
                "Value of the Main Residence",
                "Value of other Properties",
                "Wealth Other Real Assets",
                "Wealth in Deposits",
                "Wealth in Other Financial Assets",
                "Wealth in Financial Assets",
                #
                "Outstanding Balance of HMR Mortgages",
                "Outstanding Balance of Mortgages on other Properties",
                "Outstanding Balance of other Non-Mortgage Loans",
                "Debt",
                "Debt Installments",
                #
                "Tenure Status of the Main Residence",
                "Number of Properties other than Household Main Residence",
            ]
        ]

    def compute_household_wealth(self, wealth_distribution_independents: list[str]) -> None:
        self.set_household_other_real_assets_wealth()
        self.set_household_total_real_assets()
        self.set_household_deposits()
        self.set_household_other_financial_assets()
        self.set_household_financial_assets()
        self.set_household_wealth()
        self.set_household_mortgage_debt()
        self.set_household_other_debt()
        self.set_household_debt()
        self.set_household_net_wealth()
        self.set_wealth_distribution_function(independents=wealth_distribution_independents)

    def compute_household_income(
        self,
        central_gov_config: dict[str, Any],
        total_social_transfers: float,
    ) -> None:
        self.set_household_social_transfers(
            independents=central_gov_config["functions"]["household_social_transfers"]["parameters"]["independents"][
                "value"
            ],
            total_social_transfers=total_social_transfers,
        )
        self.set_household_employee_income()
        self.set_household_income_from_financial_assets()
        self.set_household_income()

    def set_household_housing_data(
        self,
        rent_as_fraction_of_unemployment_rate: float,
        unemployment_benefits_by_capita: float,
    ) -> None:
        # Whether the household owns or rents
        self.household_data.loc[
            self.household_data["Tenure Status of the Main Residence"] == 2,
            "Tenure Status of the Main Residence",
        ] = 1
        self.household_data.loc[
            self.household_data["Tenure Status of the Main Residence"] == 4,
            "Tenure Status of the Main Residence",
        ] = 1
        self.household_data.loc[
            self.household_data["Tenure Status of the Main Residence"] == 3,
            "Tenure Status of the Main Residence",
        ] = 0
        households_renting = self.household_data["Tenure Status of the Main Residence"] == 0
        households_owning = self.household_data["Tenure Status of the Main Residence"] == 1

        # Rent paid and value of the household main residence
        self.household_data.loc[:, "Rent Paid"] *= self.scale
        self.household_data.loc[:, "Value of the Main Residence"] *= self.scale
        self.household_data.loc[
            np.logical_and(households_renting, self.household_data["Rent Paid"] == 0.0),
            "Rent Paid",
        ] = np.nan
        self.household_data.loc[
            np.logical_and(
                households_owning,
                self.household_data["Value of the Main Residence"] == 0.0,
            ),
            "Value of the Main Residence",
        ] = np.nan
        self.household_data.loc[
            :,
            [
                "Type",
                "Rent Paid",
                "Value of the Main Residence",
            ],
        ] = IterativeImputer().fit_transform(
            self.household_data[
                [
                    "Type",
                    "Rent Paid",
                    "Value of the Main Residence",
                ]
            ].values
        )
        social_housing_rent = rent_as_fraction_of_unemployment_rate * unemployment_benefits_by_capita
        self.household_data.loc[
            self.household_data["Rent Paid"] < social_housing_rent, "Rent Paid"
        ] = social_housing_rent
        self.household_data.loc[
            households_owning,
            "Rent Paid",
        ] = 0.0

        # Number of additional properties
        self.household_data.loc[
            self.household_data["Number of Properties other than Household Main Residence"].isna(),
            "Number of Properties other than Household Main Residence",
        ] = 0
        self.household_data.loc[:, "Number of Properties other than Household Main Residence"] = self.household_data[
            "Number of Properties other than Household Main Residence"
        ].astype(int)

        # Value of other properties
        self.household_data.loc[:, "Value of other Properties"] *= self.scale
        household_without_additional_properties = (
            self.household_data["Number of Properties other than Household Main Residence"] == 0
        )
        self.household_data.loc[household_without_additional_properties, "Value of other Properties"] = 0.0
        self.household_data.loc[
            np.logical_and(
                np.logical_not(household_without_additional_properties),
                self.household_data["Value of other Properties"] == 0.0,
            ),
            "Value of other Properties",
        ] = np.nan
        self.household_data.loc[
            :,
            [
                "Type",
                "Number of Properties other than Household Main Residence",
                "Value of the Main Residence",
                "Value of other Properties",
            ],
        ] = IterativeImputer().fit_transform(
            self.household_data[
                [
                    "Type",
                    "Number of Properties other than Household Main Residence",
                    "Value of the Main Residence",
                    "Value of other Properties",
                ]
            ].values
        )

        # Rent received
        self.household_data.loc[:, "Rental Income from Real Estate"] *= self.scale
        self.household_data.loc[:, "Rental Income from Real Estate"] /= 12.0
        self.household_data.loc[
            self.household_data["Rental Income from Real Estate"] < social_housing_rent,
            "Rental Income from Real Estate",
        ] = social_housing_rent
        self.household_data.loc[household_without_additional_properties, "Rental Income from Real Estate"] = 0.0
        self.household_data.loc[
            np.logical_and(
                np.logical_not(household_without_additional_properties),
                self.household_data["Rental Income from Real Estate"] == 0.0,
            ),
            "Rental Income from Real Estate",
        ] = np.nan
        self.household_data.loc[
            :,
            [
                "Type",
                "Value of other Properties",
                "Rental Income from Real Estate",
            ],
        ] = IterativeImputer(min_value=0.0).fit_transform(
            self.household_data[
                [
                    "Type",
                    "Value of other Properties",
                    "Rental Income from Real Estate",
                ]
            ].values
        )

    def set_household_other_real_assets_wealth(self) -> None:
        self.household_data.loc[
            self.household_data["Value of Household Vehicles"].isna(),
            "Value of Household Vehicles",
        ] = 0.0
        self.household_data.loc[
            self.household_data["Value of Household Valuables"].isna(),
            "Value of Household Valuables",
        ] = 0.0
        self.household_data.loc[
            self.household_data["Value of Self-Employment Businesses"].isna(),
            "Value of Self-Employment Businesses",
        ] = 0.0

        self.household_data["Wealth Other Real Assets"] = (
            self.household_data["Value of Household Vehicles"]
            + self.household_data["Value of Household Valuables"]
            + self.household_data["Value of Self-Employment Businesses"]
        )
        self.household_data.loc[:, "Wealth Other Real Assets"] *= self.scale

    def set_household_total_real_assets(self) -> None:
        self.household_data["Wealth in Real Assets"] = (
            self.household_data["Value of the Main Residence"]
            + self.household_data["Value of other Properties"]
            + self.household_data["Wealth Other Real Assets"]
        )

    def set_household_deposits(self) -> None:
        self.household_data.loc[self.household_data["Wealth in Deposits"].isna(), "Wealth in Deposits"] = 0.0
        self.household_data.loc[:, "Outstanding Balance of Credit Line"] = 0.0
        self.household_data.loc[:, "Outstanding Balance of Credit Card Debt"] = 0.0
        self.household_data.loc[:, "Wealth in Deposits"] *= self.scale

    def set_household_other_financial_assets(self) -> None:
        self.household_data.loc[self.household_data["Mutual Funds"].isna(), "Mutual Funds"] = 0.0
        self.household_data.loc[self.household_data["Bonds"].isna(), "Bonds"] = 0.0
        self.household_data.loc[
            self.household_data["Value of Private Businesses"].isna(),
            "Value of Private Businesses",
        ] = 0.0
        self.household_data.loc[self.household_data["Shares"].isna(), "Shares"] = 0.0
        self.household_data.loc[self.household_data["Managed Accounts"].isna(), "Managed Accounts"] = 0.0
        self.household_data.loc[
            self.household_data["Money owed to Households"].isna(),
            "Money owed to Households",
        ] = 0.0
        self.household_data.loc[self.household_data["Other Assets"].isna(), "Other Assets"] = 0.0
        self.household_data.loc[self.household_data["Voluntary Pension"].isna(), "Voluntary Pension"] = 0.0

        self.household_data["Wealth in Other Financial Assets"] = (
            self.household_data["Mutual Funds"]
            + self.household_data["Bonds"]
            + self.household_data["Value of Private Businesses"]
            + self.household_data["Shares"]
            + self.household_data["Managed Accounts"]
            + self.household_data["Money owed to Households"]
            + self.household_data["Other Assets"]
            + self.household_data["Voluntary Pension"]
        )
        self.household_data.loc[:, "Wealth in Other Financial Assets"] *= self.scale

    def set_household_financial_assets(self) -> None:
        self.household_data["Wealth in Financial Assets"] = (
            self.household_data["Wealth in Deposits"] + self.household_data["Wealth in Other Financial Assets"]
        )

    def set_household_wealth(self) -> None:
        self.household_data["Wealth"] = (
            self.household_data["Wealth in Real Assets"] + self.household_data["Wealth in Financial Assets"]
        )

    def set_household_mortgage_debt(self) -> None:
        self.household_data.loc[
            self.household_data["Outstanding Balance of HMR Mortgages"].isna(),
            "Outstanding Balance of HMR Mortgages",
        ] = 0.0
        self.household_data.loc[
            self.household_data["Outstanding Balance of Mortgages on other Properties"].isna(),
            "Outstanding Balance of Mortgages on other Properties",
        ] = 0.0
        self.household_data.loc[:, "Outstanding Balance of HMR Mortgages"] *= self.scale
        self.household_data.loc[:, "Outstanding Balance of Mortgages on other Properties"] *= self.scale

    def set_household_other_debt(self) -> None:
        self.household_data.loc[
            self.household_data["Outstanding Balance of other Non-Mortgage Loans"].isna(),
            "Outstanding Balance of other Non-Mortgage Loans",
        ] = 0.0
        self.household_data.loc[:, "Outstanding Balance of other Non-Mortgage Loans"] *= self.scale

    def set_household_debt(self) -> None:
        self.household_data["Debt"] = (
            self.household_data["Outstanding Balance of HMR Mortgages"]
            + self.household_data["Outstanding Balance of Mortgages on other Properties"]
            + self.household_data["Outstanding Balance of other Non-Mortgage Loans"]
        )

    def set_debt_installments(self, credit_market_data: pd.DataFrame) -> None:
        credit_market_data_household_loans = credit_market_data.loc[credit_market_data["loan_type"].isin([4, 5])]
        debt_installments = np.zeros(len(self.household_data))
        for household_id in range(len(self.household_data)):
            curr_loans = credit_market_data_household_loans[
                credit_market_data_household_loans["loan_recipient_id"] == household_id
            ]
            for loan_id in range(len(curr_loans)):
                debt_installments[household_id] += float(
                    curr_loans.iloc[loan_id]["loan_value"] / curr_loans.iloc[loan_id]["loan_maturity"]
                )
        self.household_data["Debt Installments"] = debt_installments

    def set_household_net_wealth(self) -> None:
        self.household_data["Net Wealth"] = self.household_data["Wealth"] - self.household_data["Debt"]

    def set_wealth_distribution_function(self, independents: list[str]) -> None:
        self.household_data["Fraction Deposits / Total Financial Wealth"] = np.divide(
            self.household_data["Wealth in Deposits"].values.astype(float),
            self.household_data["Wealth in Financial Assets"].values.astype(float),
            out=np.ones_like(self.household_data["Wealth in Deposits"].values),
            where=self.household_data["Wealth in Financial Assets"].values.astype(float) != 0.0,
        )
        _, self.wealth_distribution_model = fit_linear(
            household_data=self.household_data,
            independents=independents,
            dependent="Fraction Deposits / Total Financial Wealth",
        )

    def set_household_employee_income(self) -> None:
        self.household_data["Employee Income"] = [
            self.individual_data.loc[self.household_data["Corresponding Individuals ID"][i], "Income"].sum()
            for i in range(len(self.household_data))
        ]

    def set_household_social_transfers(
        self,
        independents: list[str],
        total_social_transfers: float,
    ) -> None:
        # Household regular social transfers and pensions, impute missing values
        self.household_data.loc[
            :,
            ["Type", "Net Wealth", "Regular Social Transfers", "Income from Pensions"],
        ] = IterativeImputer(min_value=0).fit_transform(
            self.household_data.loc[
                :,
                [
                    "Type",
                    "Net Wealth",
                    "Regular Social Transfers",
                    "Income from Pensions",
                ],
            ].values
        )

        # Aggregate
        self.household_data["Regular Social Transfers"] += self.household_data["Income from Pensions"].values

        # Social transfers for each household group
        self.household_data["Regular Social Transfers"] /= self.household_data["Regular Social Transfers"].sum()
        social_transfers, self.social_transfers_model = fit_linear(
            household_data=self.household_data,
            independents=independents,
            dependent="Regular Social Transfers",
        )
        social_transfers[social_transfers < 0] = 0.0
        self.household_data["Regular Social Transfers"] = social_transfers

        # Rescale them
        self.household_data["Regular Social Transfers"] *= (
            total_social_transfers / self.household_data["Regular Social Transfers"].sum()
        )

    def set_household_income_from_financial_assets(self) -> None:
        fa_mask = np.logical_not(np.isnan(self.household_data["Income from Financial Assets"].values.astype(float)))
        self.household_data["Income from Financial Assets"] *= self.scale
        self.coefficient_fa_income = (
            self.household_data["Income from Financial Assets"].values.astype(float)[fa_mask].sum() / 12.0
        ) / (self.household_data["Wealth in Other Financial Assets"].values.astype(float)[fa_mask]).sum()
        self.household_data["Income from Financial Assets"] = (
            self.coefficient_fa_income * self.household_data["Wealth in Other Financial Assets"].values
        )

    def set_household_income(self) -> None:
        self.household_data["Income"] = (
            self.household_data["Employee Income"]
            + self.household_data["Regular Social Transfers"]
            + self.household_data["Rental Income from Real Estate"]
            + self.household_data["Income from Financial Assets"]
        )

    def set_household_saving_rates(self, function_name: str, independents: list[str]) -> None:
        # Some obvious cleaning
        self.household_data.loc[
            self.household_data["Consumption of Consumer Goods/Services as a Share of Income"].isin(["A"]),
            "Consumption of Consumer Goods/Services as a Share of Income",
        ] = np.nan
        self.household_data.loc[:, "Consumption of Consumer Goods/Services as a Share of Income"] = self.household_data[
            "Consumption of Consumer Goods/Services as a Share of Income"
        ].astype(float)
        self.household_data.loc[
            self.household_data["Consumption of Consumer Goods/Services as a Share of Income"] > 1.0,
            "Consumption of Consumer Goods/Services as a Share of Income",
        ] = np.nan

        # Impute missing values
        temp_imp = IterativeImputer(min_value=0, max_value=1).fit_transform(
            self.household_data[
                [
                    "Type",
                    "Income",
                    "Wealth",
                    "Debt",
                    "Consumption of Consumer Goods/Services as a Share of Income",
                ]
            ].values
        )
        self.household_data.loc[:, "Saving Rate"] = 1.0 - temp_imp[:, 4]

        # Fit a model
        saving_rates, self.saving_rates_model = fit_linear(
            household_data=self.household_data,
            independents=independents,
            dependent="Saving Rate",
        )

        # Saving rates by household characteristics
        if function_name == "AverageSavingRatesSetter":
            self.household_data["Saving Rate"] = saving_rates.mean()
        else:
            self.household_data["Saving Rate"] = saving_rates


def split_array(array_to_split: np.ndarray) -> list[list[int]]:  # pragma: no cover
    result = []
    current_sum = 0
    for num in array_to_split:
        result.append(list(range(current_sum, current_sum + num)))
        current_sum += num
    return result


def sample_households(
    hfcs_households_data: pd.DataFrame, hfcs_individuals_data: pd.DataFrame, n_households: int
) -> (pd.DataFrame, pd.DataFrame):
    """
    This function samples a specified number of households from the given HFCS households data,
    and also selects the corresponding individuals from the HFCS individuals data.

    Parameters:
    hfcs_households_data (pd.DataFrame): A DataFrame containing HFCS households data.
    hfcs_individuals_data (pd.DataFrame): A DataFrame containing HFCS individuals data.
    n_households (int): The number of households to sample.

    Returns:
    tuple: A tuple containing two DataFrames. The first DataFrame contains the selected households
    and the second DataFrame contains the individuals corresponding to the selected households.

    The function works as follows:
    1. Randomly selects a number of households from the hfcs_households_data DataFrame.
    2. Selects the corresponding individuals from the hfcs_individuals_data DataFrame.
    3. Adds the list of individuals per household to the selected households DataFrame.
    4. Resets the indices of the returned DataFrames for consistency.
    """
    # Draw households at random
    household_inds = np.random.choice(
        hfcs_households_data.shape[0],
        n_households,
        p=hfcs_households_data["Weight"] / hfcs_households_data["Weight"].sum(),
        replace=True,
    )
    household_selection = hfcs_households_data.iloc[household_inds]
    individual_selection = hfcs_individuals_data.loc[
        hfcs_individuals_data["Corresponding Household ID"].isin(household_selection.index)
    ]
    n_individuals_per_household = individual_selection.groupby("Corresponding Household ID")["Gender"].count()
    household_selection["n_individuals"] = n_individuals_per_household
    hh_ind_to_numerical = dict(zip(household_selection.index, range(len(household_selection))))
    individual_selection["Corresponding Household ID"] = individual_selection["Corresponding Household ID"].map(
        hh_ind_to_numerical
    )
    household_selection["Corresponding Individuals ID"] = split_array(household_selection["n_individuals"].values)
    household_selection.drop(["Weight", "n_individuals"], axis=1, inplace=True)
    household_selection.reset_index(inplace=True, drop=True)
    individual_selection.reset_index(inplace=True, drop=True)
    return household_selection, individual_selection
