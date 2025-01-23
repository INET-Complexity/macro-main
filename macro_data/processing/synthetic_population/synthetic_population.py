from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

RESTRICT_COLS = [
    "Type",
    "Corresponding Individuals ID",
    "Corresponding Bank ID",
    "Corresponding Inhabited House ID",
    "Corresponding Renters",
    "Corresponding Property Owner",
    "Corresponding Additionally Owned Houses ID",
    "Income",
    "Employee Income",
    "Regular Social Transfers",
    "Rental Income from Real Estate",
    "Income from Financial Assets",
    "Saving Rate",
    "Rent Paid",
    "Rent Imputed",
    "Wealth",
    "Net Wealth",
    "Wealth in Real Assets",
    "Value of the Main Residence",
    "Value of other Properties",
    "Wealth Other Real Assets",
    "Wealth in Deposits",
    "Wealth in Other Financial Assets",
    "Wealth in Financial Assets",
    "Outstanding Balance of HMR Mortgages",
    "Outstanding Balance of Mortgages on other Properties",
    "Outstanding Balance of other Non-Mortgage Loans",
    "Debt",
    "Debt Installments",
    "Tenure Status of the Main Residence",
    "Number of Properties other than Household Main Residence",
]


class SyntheticPopulation(ABC):
    """
    Represents a synthetic population for a specific country and year.

    The household data is a pandas data frame with the following columns:
        - Type: The type of the household (1: single, 2: couple, 3: single parent, 4: couple with children).
        - Corresponding Individuals ID: The IDs of the individuals in the household.
        - Corresponding Bank ID: The ID of the bank the household is associated with.
        - Corresponding Inhabited House ID: The ID of the house the household inhabits.
        - Corresponding Renters: The IDs of the individuals in the household who rent.
        - Corresponding Property Owner: The IDs of the individuals in the household who own property.
        - Corresponding Additionally Owned Houses ID: The IDs of the houses the household owns.
        - Income: The total income of the household.
        - Employee Income: The income of the household from employment.
        - Regular Social Transfers: The income of the household from social transfers.
        - Rental Income from Real Estate: The income of the household from rental of real estate.
        - Income from Financial Assets: The income of the household from financial assets.
        - Saving Rate: The saving rate of the household.
        - Rent Paid: The rent paid by the household.
        - Rent Imputed: The imputed rent of the household.
        - Wealth: The total wealth of the household.
        - Net Wealth: The net wealth of the household.
        - Wealth in Real Assets: The wealth of the household in real assets.
        - Value of the Main Residence: The value of the main residence of the household.
        - Value of other Properties: The value of other properties of the household.
        - Wealth Other Real Assets: The wealth of the household in other real assets.
        - Wealth in Deposits: The wealth of the household in deposits.
        - Wealth in Other Financial Assets: The wealth of the household in other financial assets.
        - Wealth in Financial Assets: The wealth of the household in financial assets.
        - Outstanding Balance of HMR Mortgages: The outstanding balance of the household's HMR mortgages.
        - Outstanding Balance of Mortgages on other Properties: The outstanding balance of the household's mortgages on other properties.
        - Outstanding Balance of other Non-Mortgage Loans: The outstanding balance of the household's other non-mortgage loans.
        - Debt: The total debt of the household.
        - Debt Installments: The debt installments of the household (monthly payments of debt).
        - Tenure Status of the Main Residence: The tenure status of the main residence of the household.
        - Number of Properties other than Household Main Residence: The number of properties other than the household's main residence.

    The individual data is a pandas data frame with the following columns:
        - Gender: The gender of the individual (1: male, 2: female)
        - Age: The age of the individual.
        - Education: The education level of the individual (ISCED classification).
        - Activity Status: The activity status of the individual (1: employed, 2: unemployed, 3: not economically active).
        - Employment Industry: The industry of the individual's employment.
        - Employee Income: The income of the individual from employment.
        - Income from Unemployment Benefits: The income of the individual from unemployment benefits.
        - Income: The total income of the individual.
        - Corresponding Household ID: The ID of the household the individual belongs to.
        - Corresponding Firm ID: The ID of the firm the individual works for.

    Attributes:
        country_name (str): The name of the country.
        country_name_short (str): The short name or code of the country.
        scale (int): The scale of the synthetic population.
        year (int): The year of the synthetic population.
        industries (list[str]): The list of industries in the country.
        individual_data (pd.DataFrame): The data frame containing individual-level data.
        household_data (pd.DataFrame): The data frame containing household-level data.
        social_housing_rent (float): The rent for social housing.
        coefficient_fa_income (float): The coefficient for family allowance income.
        consumption_weights (np.ndarray): The weights for household consumption.
        consumption_weights_by_income (np.ndarray): The weights for household consumption based on income.
        saving_rates_model (LinearRegression): The model for household saving rates.
        social_transfers_model (LinearRegression): The model for social transfers.
        wealth_distribution_model (LinearRegression): The model for wealth distribution.
    """

    @abstractmethod
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
        investment: np.ndarray,
        saving_rates_model: LinearRegression,
        social_transfers_model: LinearRegression,
        wealth_distribution_model: LinearRegression,
        yearly_factor: float = 4.0,
    ):
        self.country_name = country_name
        self.country_name_short = country_name_short
        self.scale = scale
        self.year = year
        self.industries = industries

        # Agents data
        self.individual_data = individual_data
        self.household_data = household_data

        # Convenience
        self.social_housing_rent = social_housing_rent
        self.coefficient_fa_income = coefficient_fa_income

        # Household consumption weights and models
        self.consumption_weights = consumption_weights
        self.consumption_weights_by_income = consumption_weights_by_income
        self.investment = investment

        self.saving_rates_model = saving_rates_model
        self.social_transfers_model = social_transfers_model
        self.wealth_distribution_model = wealth_distribution_model
        self.yearly_factor = yearly_factor

    def set_individual_labour_inputs(
        self,
        firm_production: np.ndarray,
        firm_employees: pd.DataFrame,
        unemployment_labour_inputs_fraction: float = 0.3,
        override: bool = True,
    ) -> None:
        if override:
            self.individual_data["Labour Inputs"] = 1.0
        else:
            self.individual_data["Labour Inputs"] = np.nan

            # Employed individuals contribute labour inputs proportional to their income from employment
            for firm_id in range(firm_production.shape[0]):
                self.individual_data.loc[firm_employees[firm_id], "Labour Inputs"] = (
                    self.individual_data.loc[firm_employees[firm_id], "Employee Income"]
                    / self.individual_data.loc[firm_employees[firm_id], "Employee Income"].sum()
                    * firm_production[firm_id]
                )

            # Unemployed individuals initial labour inputs are set to be a fraction of the
            # mean labour inputs in the industry
            for industry in range(len(self.industries)):
                self.individual_data.loc[
                    np.logical_and(
                        self.individual_data["Activity Status"] == 2,
                        self.individual_data["Employment Industry"] == industry,
                    ),
                    "Labour Inputs",
                ] = unemployment_labour_inputs_fraction * np.mean(
                    self.individual_data.loc[
                        np.logical_and(
                            self.individual_data["Activity Status"] == 1,
                            self.individual_data["Employment Industry"] == industry,
                        ),
                        "Labour Inputs",
                    ]
                )

            # Not economically active individuals contribute no labour inputs
            self.individual_data.loc[
                self.individual_data["Activity Status"] == 3,
                "Labour Inputs",
            ] = 0.0

    @property
    def industry_consumption_before_vat(self):
        return ...

    @property
    def investment_weights(self) -> np.ndarray:
        """
        Returns the investment weights.

        Returns:
            np.ndarray: The investment weights.
        """
        return self.investment / self.investment.sum()

    @abstractmethod
    def compute_household_income(
        self,
        total_social_transfers: float,
        independents: Optional[list[str]] = None,
    ) -> None: ...

    @property
    def number_of_households(self):
        return self.household_data.shape[0]

    @property
    def number_employees_by_industry(self) -> np.ndarray:
        number_employees_by_industry = np.zeros(len(self.industries))
        for industry_ind in range(len(self.industries)):
            number_employees_by_industry[industry_ind] = int(
                np.sum(
                    np.logical_and(
                        self.individual_data["Employment Industry"] == industry_ind,
                        self.individual_data["Activity Status"] == 1,
                    )
                )
            )
        return number_employees_by_industry.astype(int)

    def set_consumption_weights(self, consumption_weights: np.ndarray) -> None:
        self.consumption_weights = consumption_weights.copy()

    @abstractmethod
    def set_debt_installments(
        self, consumption_installments: np.ndarray, ce_installments: np.ndarray, mortgage_installments: np.ndarray
    ) -> None: ...

    @abstractmethod
    def set_household_saving_rates(self, independents: Optional[list[str]] = None) -> None: ...

    @abstractmethod
    def compute_household_wealth(self, independents: Optional[list[str]] = None) -> None: ...

    def set_income(self) -> None:
        self.individual_data["Income"] = (
            self.individual_data["Employee Income"].fillna(0.0).values
            + self.individual_data["Income from Unemployment Benefits"].fillna(0.0).values
        )

    def restrict(self): ...

    def normalise_household_consumption(
        self,
        iot_hh_consumption: np.ndarray | pd.Series,
        vat: float,
        positive_saving_rates_only: bool = True,
        independents: Optional[list[str]] = None,
    ): ...

    def set_household_investment_rates(
        self,
        capital_formation_taxrate: float,
        default_investment_rates: np.ndarray | float = 0.2,
    ) -> None: ...

    def normalise_household_investment(
        self, tau_cf: float, iot_hh_investment: np.ndarray | pd.Series, positive_investment_rates: bool = True
    ) -> None:
        inv_weights = iot_hh_investment / iot_hh_investment.sum()
        income = self.household_data["Income"].values
        investment_rate = self.household_data["Investment Rate"].values

        current_hh_investment = default_target_investment(
            income_=income, investment_rate=investment_rate, tau_cf_=tau_cf, investment_weights_=inv_weights
        )

        factor = iot_hh_investment.sum() / current_hh_investment.sum()

        self.household_data["Investment Rate"] = factor * investment_rate

        self.investment *= factor

        # set initial investment
        self.household_data["Investment"] = (self.household_data["Income"] * self.household_data["Investment Rate"]) / (
            1 + tau_cf
        )

    def get_current_hh_investment_by_industry(self, tau_cf: float) -> np.ndarray:
        income = self.household_data["Income"].values
        investment_rate = self.household_data["Investment Rate"].values
        return default_target_investment(
            income_=income, investment_rate=investment_rate, tau_cf_=tau_cf, investment_weights_=self.investment_weights
        )

    def match_consumption_weights_by_income(
        self,
        weights_by_income: np.ndarray | pd.DataFrame,
        iot_hh_consumption: pd.Series,
        vat: float,
        consumption_variance: float = 0.1,
    ) -> None: ...

    def set_wealth_distribution_function(self, independents: Optional[list[str]] = None) -> None: ...

    def add_emissions(
        self, emission_factors_array: np.ndarray, emitting_indices: list[int] | np.ndarray, tau_cf: float
    ) -> None:
        consumption_emissions = self.industry_consumption_before_vat[:, emitting_indices] @ emission_factors_array
        self.household_data["Consumption Emissions"] = consumption_emissions

        investment_emissions = (
            self.get_current_hh_investment_by_industry(tau_cf)[:, emitting_indices] @ emission_factors_array
        )

        self.household_data["Investment Emissions"] = investment_emissions

        # decompose in oil, gas, coal and refined products emissions

        for i, name in enumerate(["Coal", "Gas", "Oil", "Refined Products"]):
            self.household_data[f"{name} Consumption Emissions"] = (
                self.industry_consumption_before_vat[:, emitting_indices[i]] * emission_factors_array[i]
            )
            # investment
            self.household_data[f"{name} Investment Emissions"] = (
                self.get_current_hh_investment_by_industry(tau_cf)[:, emitting_indices[i]] * emission_factors_array[i]
            )

    @property
    def total_emissions(self) -> float:
        return self.household_data["Consumption Emissions"] + self.household_data["Investment Emissions"]


def default_target_investment(
    income_: np.ndarray,
    investment_weights_: np.ndarray,
    investment_rate: np.ndarray,
    tau_cf_: float,
) -> np.ndarray:
    return 1.0 / (1 + tau_cf_) * np.outer(investment_weights_, investment_rate * income_).T
