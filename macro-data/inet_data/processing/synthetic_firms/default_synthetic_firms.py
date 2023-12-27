import logging

import numpy as np
import pandas as pd

from inet_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks
from inet_data.processing.synthetic_credit_market.synthetic_credit_market import SyntheticCreditMarket
from inet_data.processing.synthetic_firms.firm_tools import (
    initialise_basic_firm_fields,
    function_parameters_dependent_initialisation,
)
from inet_data.processing.synthetic_firms.synthetic_firms import SyntheticFirms
from inet_data.readers.default_readers import DataReaders


class SyntheticDefaultFirms(SyntheticFirms):
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
        super().__init__(
            country_name,
            scale,
            year,
            industries,
            number_of_firms_by_industry,
            firm_data,
            intermediate_inputs_stock,
            used_intermediate_inputs,
            capital_inputs_stock,
            used_capital_inputs,
            total_firm_deposits,
            total_firm_debt,
            capital_inputs_productivity_matrix,
            intermediate_inputs_productivity_matrix,
            capital_inputs_depreciation_matrix,
        )

    @classmethod
    def init_from_readers(
        cls,
        readers: DataReaders,
        country_name: str,
        year: int,
        industries: list[str],
        scale: int,
        industry_data: dict[str, pd.DataFrame],
        n_employees_per_industry: np.ndarray,
        assume_zero_initial_deposits: bool,
        assume_zero_initial_debt: bool,
        initial_inventory_to_input_fraction: float = 0,
        intermediate_inputs_utilisation_rate: float = 1.0,
        capital_inputs_utilisation_rate: float = 1.0,
    ):
        n_firms_per_industry = industry_data["industry_vectors"]["Number of Firms"].values
        n_firms = n_firms_per_industry.sum()

        firm_data = pd.DataFrame(index=range(n_firms))
        firm_size_zetas = readers.oecd_econ.read_firm_size_zetas(
            country_name,
            year,
        )
        if firm_size_zetas is None:
            firm_size_zetas = readers.ons.get_firm_size_zetas()
        exchange_rate = readers.exchange_rates.from_usd_to_lcu(country_name, year)
        tau_sif = readers.oecd_econ.read_tau_sif(country_name, year)

        firm_data = initialise_basic_firm_fields(
            firm_data,
            industry_data,
            n_employees_per_industry,
            n_firms_per_industry,
            firm_size_zetas,
            exchange_rate,
            tau_sif,
        )

        total_firm_deposits = readers.eurostat.get_total_nonfin_firm_deposits(country_name, year)
        total_firm_debt = readers.eurostat.get_total_nonfin_firm_debt(country_name, year)

        capital_inputs_productivity_matrix = industry_data["capital_inputs_productivity_matrix"].values
        intermediate_inputs_productivity_matrix = industry_data["intermediate_inputs_productivity_matrix"].values
        capital_inputs_depreciation_matrix = industry_data["capital_inputs_depreciation_matrix"].values
        (
            capital_inputs_stock,
            intermediate_inputs_stock,
            used_capital_inputs,
            used_intermediate_inputs,
        ) = function_parameters_dependent_initialisation(
            firm_data,
            intermediate_inputs_productivity_matrix,
            capital_inputs_depreciation_matrix,
            capital_inputs_productivity_matrix,
            total_firm_deposits,
            total_firm_debt,
            assume_zero_initial_debt,
            assume_zero_initial_deposits,
            capital_inputs_utilisation_rate,
            initial_inventory_to_input_fraction,
            intermediate_inputs_utilisation_rate,
        )

        firm_data["Employees ID"] = [[] for _ in range(n_firms)]

        return cls(
            country_name=country_name,
            scale=scale,
            year=year,
            industries=industries,
            number_of_firms_by_industry=n_firms_per_industry,
            firm_data=firm_data,
            intermediate_inputs_stock=intermediate_inputs_stock,
            used_intermediate_inputs=used_intermediate_inputs,
            capital_inputs_stock=capital_inputs_stock,
            used_capital_inputs=used_capital_inputs,
            total_firm_deposits=total_firm_deposits,
            total_firm_debt=total_firm_debt,
            capital_inputs_productivity_matrix=capital_inputs_productivity_matrix,
            intermediate_inputs_productivity_matrix=intermediate_inputs_productivity_matrix,
            capital_inputs_depreciation_matrix=capital_inputs_depreciation_matrix,
        )

    ### THINGS TO BE SET AFTER MATCHING
    def set_taxes_paid_on_production(self, taxes_less_subsidies_rates: np.ndarray) -> None:
        self.firm_data["Taxes paid on Production"] = (
            taxes_less_subsidies_rates[self.firm_data["Industry"].values]
            * self.firm_data["Production"].values
            * self.firm_data["Price"].values
        )

    def set_interest_paid(
        self,
        interest_rate_on_firm_deposits: np.ndarray,
        overdraft_rate_on_firm_deposits: np.ndarray,
        credit_market_data: pd.DataFrame,
    ) -> None:
        # Interest on deposits
        self.firm_data["Interest paid on deposits"] = -interest_rate_on_firm_deposits[
            self.firm_data["Corresponding Bank ID"].values
        ] * np.maximum(0.0, self.firm_data["Deposits"].values) - overdraft_rate_on_firm_deposits[
            self.firm_data["Corresponding Bank ID"].values
        ] * np.minimum(
            0.0, self.firm_data["Deposits"].values
        )

        # Interest paid on loans
        credit_market_data_firm_loans = credit_market_data.loc[credit_market_data["loan_type"] == 2]
        interest_on_loans = np.zeros(len(self.firm_data))
        for firm_id in range(len(self.firm_data)):
            curr_loans = credit_market_data_firm_loans[credit_market_data_firm_loans["loan_recipient_id"] == firm_id]
            for loan_id in range(len(curr_loans)):
                interest_on_loans[firm_id] += float(
                    curr_loans.iloc[loan_id]["loan_interest_rate"] * curr_loans.iloc[loan_id]["loan_value"]
                )
        self.firm_data["Interest paid on loans"] = interest_on_loans

        # Total interest paid
        self.firm_data["Interest paid"] = (
            self.firm_data["Interest paid on deposits"] + self.firm_data["Interest paid on loans"]
        )

    def set_firm_profits(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        tau_sif: float,
    ) -> None:
        # Sales
        sales = self.firm_data["Production"].values * self.firm_data["Price"].values

        # Labour
        labour_costs = self.firm_data["Total Wages"].values * (1 + tau_sif)

        # Intermediate inputs
        intermediate_inputs_costs = (
            self.firm_data["Production"].values
            / intermediate_inputs_productivity_matrix[:, self.firm_data["Industry"].values]
        ).T.sum(axis=1) * self.firm_data["Price"].values

        # Capital inputs
        capital_inputs_costs = (
            self.firm_data["Production"].values
            * capital_inputs_depreciation_matrix[:, self.firm_data["Industry"].values]
        ).T.sum(axis=1) * self.firm_data["Price"].values

        # Update profits
        self.firm_data["Profits"] = (
            sales
            - labour_costs
            - intermediate_inputs_costs
            - capital_inputs_costs
            - self.firm_data["Taxes paid on Production"].values
            - self.firm_data["Interest paid"].values
        )
        logging.info(f"Initial profits: {self.firm_data['Profits']}")

    def set_unit_costs(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        tau_sif: float,
    ) -> None:
        # Labour
        labour_costs = self.firm_data["Total Wages"].values * (1 + tau_sif)

        # Intermediate inputs
        intermediate_inputs_costs = (
            self.firm_data["Production"].values[:, None]
            / intermediate_inputs_productivity_matrix[:, self.firm_data["Industry"].values].T
        ).sum(axis=1) * self.firm_data["Price"].values

        # Capital inputs
        capital_inputs_costs = (
            self.firm_data["Production"].values[:, None]
            * capital_inputs_depreciation_matrix[:, self.firm_data["Industry"].values].T
        ).sum(axis=1) * self.firm_data["Price"].values

        # Update unit costs
        self.firm_data["Unit Costs"] = (
            labour_costs
            + intermediate_inputs_costs
            + capital_inputs_costs
            + self.firm_data["Taxes paid on Production"].values
        ) / self.firm_data["Production"].values

    def set_corporate_taxes_paid(self, tau_firm: float) -> None:
        self.firm_data["Corporate Taxes Paid"] = tau_firm * np.maximum(0.0, self.firm_data["Profits"])

    def set_firm_debt_installments(self, synthetic_credit_market: SyntheticCreditMarket) -> None:
        credit_market_data = synthetic_credit_market.credit_market_data
        credit_market_data_firm_loans = credit_market_data.loc[credit_market_data["loan_type"] == 2]
        debt_installments = np.zeros(len(self.firm_data))
        for firm_id in range(len(self.firm_data)):
            curr_loans = credit_market_data_firm_loans[credit_market_data_firm_loans["loan_recipient_id"] == firm_id]
            for loan_id in range(len(curr_loans)):
                debt_installments[firm_id] += float(
                    curr_loans.iloc[loan_id]["loan_value"] / curr_loans.iloc[loan_id]["loan_maturity"]
                )
        self.firm_data["Debt Installments"] = debt_installments

    def set_additional_initial_conditions(
        self,
        readers: DataReaders,
        industry_data: dict[str, pd.DataFrame],
        synthetic_banks: SyntheticBanks,
        synthetic_credit_market: SyntheticCreditMarket,
    ) -> None:
        taxes_less_subsidies_rates = industry_data["industry_vectors"]["Taxes Less Subsidies Rates"].values
        interest_rate_on_firm_deposits = synthetic_banks.bank_data["Interest Rates on Firm Deposits"].values
        overdraft_rate_on_firm_deposits = synthetic_banks.bank_data["Overdraft Rate on Firm Deposits"].values

        self.set_taxes_paid_on_production(
            taxes_less_subsidies_rates=taxes_less_subsidies_rates,
        )
        self.set_interest_paid(
            interest_rate_on_firm_deposits=interest_rate_on_firm_deposits,
            overdraft_rate_on_firm_deposits=overdraft_rate_on_firm_deposits,
            credit_market_data=synthetic_credit_market.credit_market_data,
        )

        self.set_firm_profits(
            intermediate_inputs_productivity_matrix=industry_data["intermediate_inputs_productivity_matrix"].values,
            capital_inputs_depreciation_matrix=industry_data["capital_inputs_depreciation_matrix"].values,
            tau_sif=readers.oecd_econ.read_tau_sif(self.country_name, self.year),
        )

        self.set_unit_costs(
            intermediate_inputs_productivity_matrix=industry_data["intermediate_inputs_productivity_matrix"].values,
            capital_inputs_depreciation_matrix=industry_data["capital_inputs_depreciation_matrix"].values,
            tau_sif=readers.oecd_econ.read_tau_sif(self.country_name, self.year),
        )

        self.set_corporate_taxes_paid(
            tau_firm=readers.oecd_econ.read_tau_firm(self.country_name, self.year),
        )

        self.set_firm_debt_installments(
            synthetic_credit_market=synthetic_credit_market,
        )
