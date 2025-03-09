"""Module for preprocessing synthetic firm data.

This module provides a framework for preprocessing and organizing firm-level data
that will be used to initialize behavioral models. Key preprocessing includes:

1. Firm Data Organization:
   - Industry classification
   - Employee allocation
   - Financial data processing
   - Production parameters

2. Initial State Processing:
   - Balance sheet data
   - Production capacity
   - Input requirements
   - Financial positions

3. Parameter Processing:
   - Productivity metrics
   - Cost structures
   - Tax parameters
   - Interest rates

Note:
    This module is NOT used for simulating firm behavior. It only handles
    the preprocessing and organization of firm data that will later be used
    to initialize behavioral models in the simulation package.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from macro_data.processing.country_data import TaxData
from macro_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks
from macro_data.processing.synthetic_credit_market.loan_data import (
    LongtermLoans,
    ShorttermLoans,
)


class SyntheticFirms(ABC):
    """Container for preprocessed firm-level data.

    The firm data is stored in a pandas DataFrame with the following columns:
        - Industry: The industry of the firm.
        - Number of Employees: The number of employees of the firm.
        - Total Wages: The total wages of the firm.
        - Total Wages Paid: The total wages paid by the firm.
        - Production: The production of the firm (in LCU).
        - Price in USD: The price of the firm (in USD).
        - Price: The price of the firm (in LCU).
        - Labour Inputs: The labour inputs of the firm (in LCU).
        - Inventory: The inventory of the firm (in LCU).
        - Demand: The demand of the firm (in LCU).
        - Deposits: The deposits of the firm (in LCU).
        - Debt: The debt of the firm (in LCU).
        - Equity: The equity of the firm (in LCU).
        - Employees ID: The IDs of the employees of the firm.
        - Corresponding Bank ID: The ID of the corresponding bank of the firm.
        - Taxes paid on Production: The taxes paid on production of the firm (in LCU).
        - Interest paid on deposits: The interest paid on deposits of the firm (in LCU).
        - Interest paid on loans: The interest paid on loans of the firm (in LCU).
        - Interest paid: The interest paid of the firm (in LCU).
        - Profits: The profits of the firm (in LCU).
        - Unit Costs: The unit costs of the firm (in LCU).
        - Corporate Taxes Paid: The corporate taxes paid of the firm (in LCU).
        - Debt Installments: The debt installments of the firm (in LCU).

    Attributes:
        country_name (str): Country identifier for data collection
        scale (int): Scaling factor for synthetic data
        year (int): Reference year for preprocessing
        industries (list[str]): List of industry classifications
        number_of_firms_by_industry (np.ndarray): Firm count by industry
        firm_data (pd.DataFrame): Main data container with firm information
        intermediate_inputs_stock (np.ndarray): Initial intermediate input inventory
        used_intermediate_inputs (np.ndarray): Initial input usage
        capital_inputs_stock (np.ndarray): Initial capital stock
        used_capital_inputs (np.ndarray): Initial capital usage
        total_firm_deposits (float): Aggregate firm deposits
        total_firm_debt (float): Aggregate firm debt
        capital_inputs_productivity_matrix (np.ndarray): Capital productivity parameters
        intermediate_inputs_productivity_matrix (np.ndarray): Input productivity parameters
        capital_inputs_depreciation_matrix (np.ndarray): Capital depreciation rates
        labour_productivity_by_industry (np.ndarray): Labor productivity by industry
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
        labour_productivity_by_industry: np.ndarray,
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

        self.labour_productivity_by_industry = labour_productivity_by_industry

    @property
    def number_of_firms(self) -> int:
        """
        Returns the total number of firms.

        Returns:
            int: The total number of firms.
        """
        return self.number_of_firms_by_industry.sum()

    def reset_function_parameters(
        self,
        capital_inputs_utilisation_rate: float,
        initial_inventory_to_input_fraction: float,
        intermediate_inputs_utilisation_rate: float,
        zero_initial_debt: bool,
        zero_initial_deposits: bool,
    ):
        pass

    def set_additional_initial_conditions(
        self,
        industry_data: dict[str, pd.DataFrame],
        synthetic_banks: SyntheticBanks,
        long_term_loans: LongtermLoans,
        short_term_loans: ShorttermLoans,
        tax_data: TaxData,
    ) -> None: ...

    @property
    def total_emissions(self):
        return self.firm_data["Input Emissions"] + self.firm_data["Capital Emissions"]
