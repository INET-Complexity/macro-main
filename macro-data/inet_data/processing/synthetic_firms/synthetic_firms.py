from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class SyntheticFirms(ABC):
    """
    Represents a synthetic firms object.

    Args:
        country_name (str): The name of the country.
        scale (int): The scale of the synthetic firms.
        year (int): The year of the synthetic firms.
        industries (list[str]): The list of industries.
        number_of_firms_by_industry (np.ndarray): The number of firms by industry.
        firm_data (pd.DataFrame): The firm data.
        intermediate_inputs_stock (np.ndarray): The intermediate inputs stock.
        used_intermediate_inputs (np.ndarray): The used intermediate inputs.
        capital_inputs_stock (np.ndarray): The capital inputs stock.
        used_capital_inputs (np.ndarray): The used capital inputs.
        total_firm_deposits (float): The total firm deposits.
        total_firm_debt (float): The total firm debt.
        capital_inputs_productivity_matrix (np.ndarray): The capital inputs productivity matrix.
        intermediate_inputs_productivity_matrix (np.ndarray): The intermediate inputs productivity matrix.
        capital_inputs_depreciation_matrix (np.ndarray): The capital inputs depreciation matrix.
    """

    @abstractmethod
    def __init__(
        self,
        country_name: str,
        scale: int,
        year: int,
        industries: list[str],
        number_of_firms_by_industry: np.ndarray,
        firm_data: pd.DataFrame,
        intermediate_inputs_stock: np.ndarray,
        used_intermediate_inputs: np.ndarray,
        capital_inputs_stock: np.ndarray,
        used_capital_inputs: np.ndarray,
        total_firm_deposits: float,
        total_firm_debt: float,
        capital_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_productivity_matrix: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
    ):
        self.country_name = country_name
        self.scale = scale
        self.year = year
        self.industries = industries

        # Firm data
        self.number_of_firms_by_industry = number_of_firms_by_industry
        self.firm_data = firm_data

        # WILL BE DEPRECATED
        self.intermediate_inputs_stock = intermediate_inputs_stock
        self.used_intermediate_inputs = used_intermediate_inputs
        self.capital_inputs_stock = capital_inputs_stock
        self.used_capital_inputs = used_capital_inputs

        # New fields
        self.total_firm_deposits = total_firm_deposits
        self.total_firm_debt = total_firm_debt
        self.capital_inputs_productivity_matrix = capital_inputs_productivity_matrix
        self.intermediate_inputs_productivity_matrix = intermediate_inputs_productivity_matrix
        self.capital_inputs_depreciation_matrix = capital_inputs_depreciation_matrix

    @property
    def number_of_firms(self) -> int:
        """
        Returns the total number of firms.

        Returns:
            int: The total number of firms.
        """
        return self.number_of_firms_by_industry.sum()
