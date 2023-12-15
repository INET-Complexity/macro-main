from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from inet_data.readers.economic_data.oecd_economic_data import OECDEconData


class SyntheticFirms(ABC):
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
        return self.number_of_firms_by_industry.sum()

    def set_additional_initial_conditions(
        self,
        econ_reader: OECDEconData,
        industry_data: dict[str, pd.DataFrame],
        interest_rate_on_firm_deposits: np.ndarray,
        overdraft_rate_on_firm_deposits: np.ndarray,
        credit_market_data: pd.DataFrame,
    ) -> None:
        ...
        # self.set_taxes_paid_on_production(
        #     taxes_less_subsidies_rates=industry_data["industry_vectors"]["Taxes Less Subsidies Rates"].values,
        # )
        # self.set_interest_paid(
        #     interest_rate_on_firm_deposits=interest_rate_on_firm_deposits,
        #     overdraft_rate_on_firm_deposits=overdraft_rate_on_firm_deposits,
        #     credit_market_data=credit_market_data,
        # )
        # self.set_firm_profits(
        #     intermediate_inputs_productivity_matrix=industry_data["intermediate_inputs_productivity_matrix"].values,
        #     capital_inputs_depreciation_matrix=industry_data["capital_inputs_depreciation_matrix"].values,
        #     tau_sif=econ_reader.read_tau_sif(self.country_name, self.year),
        # )
        # self.set_unit_costs(
        #     intermediate_inputs_productivity_matrix=industry_data["intermediate_inputs_productivity_matrix"].values,
        #     capital_inputs_depreciation_matrix=industry_data["capital_inputs_depreciation_matrix"].values,
        #     tau_sif=econ_reader.read_tau_sif(self.country_name, self.year),
        # )
        # self.set_corporate_taxes_paid(
        #     tau_firm=econ_reader.read_tau_firm(self.country_name, self.year),
        # )
        # self.set_firm_debt_installments(credit_market_data=credit_market_data)
