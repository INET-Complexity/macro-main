from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class SyntheticPopulation(ABC):
    """
    Represents a synthetic population for a specific country and year.

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
        saving_rates_model: LinearRegression,
        social_transfers_model: LinearRegression,
        wealth_distribution_model: LinearRegression,
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
        self.saving_rates_model = saving_rates_model
        self.social_transfers_model = social_transfers_model
        self.wealth_distribution_model = wealth_distribution_model

    def set_individual_labour_inputs(
        self,
        firm_production: np.ndarray,
        firm_employees: pd.DataFrame,
        unemployment_labour_inputs_fraction: float = 0.3,
    ) -> None:
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

    @abstractmethod
    def compute_household_income(
        self,
        total_social_transfers: float,
    ) -> None:
        pass

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
    def set_debt_installments(self, credit_market_data: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def set_household_saving_rates(
        self, function_name: str = "AverageSavingRatesSetter", independents: Optional[list[str]] = None
    ) -> None:
        pass

    @abstractmethod
    def compute_household_wealth(self) -> None:
        ...

    def set_income(self) -> None:
        self.individual_data["Income"] = (
            self.individual_data["Employee Income"].fillna(0.0).values
            + self.individual_data["Income from Unemployment Benefits"].fillna(0.0).values
        )
