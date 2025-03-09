"""Module for preprocessing default synthetic firm data.

This module provides a concrete implementation for preprocessing firm-level data
using standard data sources (OECD, Eurostat, Compustat). Key preprocessing includes:

1. Data Source Integration:
   - OECD economic indicators
   - Eurostat business statistics
   - Compustat firm-level data
   - National accounts data

2. Initial State Processing:
   - Industry-level aggregates
   - Firm size distributions
   - Financial positions
   - Production parameters

3. Parameter Estimation:
   - Productivity metrics
   - Input-output relationships
   - Tax rates
   - Interest rates

Note:
    This module is NOT used for simulating firm behavior. It only handles
    the preprocessing and organization of data from standard sources that will
    later be used to initialize behavioral models in the simulation package.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.configuration.dataconfiguration import FirmsDataConfiguration
from macro_data.processing.country_data import TaxData
from macro_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks
from macro_data.processing.synthetic_credit_market.loan_data import (
    LongtermLoans,
    ShorttermLoans,
)
from macro_data.processing.synthetic_firms.firm_tools import (
    function_parameters_dependent_initialisation,
    initialise_basic_firm_fields,
    initialise_basic_firm_fields_compustat,
)
from macro_data.processing.synthetic_firms.synthetic_firms import SyntheticFirms
from macro_data.readers.default_readers import DataReaders
from macro_data.readers.emissions.emissions_reader import EmissionsData


class DefaultSyntheticFirms(SyntheticFirms):
    """Container for preprocessed firm data using standard data sources.

    This class provides a concrete implementation for preprocessing firm-level data
    using standard data sources (OECD, Eurostat, Compustat). It processes and
    organizes data about firms' characteristics, financial positions, and production
    parameters. It does NOT implement any firm behavior - it only handles data
    preprocessing.

    The preprocessing workflow includes:
    1. Data Collection:
       - Reading from standard data sources
       - Handling missing data
       - Currency conversion
       - Scale adjustment

    2. Firm Structure:
       - Industry classification
       - Size distribution estimation
       - Employee allocation
       - Bank relationship mapping

    3. Financial Processing:
       - Balance sheet construction
       - Income statement elements
       - Tax calculations
       - Interest computations

    4. Production Parameters:
       - Input requirements
       - Productivity metrics
       - Cost structures
       - Initial inventory levels

    Note:
        This is a data container class. The actual firm behavior is implemented
        in the simulation package, which uses this preprocessed data for
        initialization.

    Attributes:
        Inherits all attributes from SyntheticFirms base class.
    """

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
            labour_productivity_by_industry,
        )

    @classmethod
    def from_readers(
        cls,
        readers: DataReaders,
        country_name: Country,
        year: int,
        industries: list[str],
        scale: int,
        industry_data: dict[str, pd.DataFrame],
        n_employees_per_industry: np.ndarray,
        firm_configuration: FirmsDataConfiguration,
        exchange_rate_from_eur: float = 1.0,
        proxy_country: Optional[Country] = None,
        emission_factors: Optional[EmissionsData] = None,
    ) -> "DefaultSyntheticFirms":
        n_firms_per_industry = industry_data["industry_vectors"]["Number of Firms"].values
        # number of firms per industry is at most the number of employees per industry
        n_firms_per_industry = np.minimum(n_firms_per_industry, n_employees_per_industry)
        n_firms = n_firms_per_industry.sum()

        firm_data = pd.DataFrame(index=range(n_firms))

        exchange_rate = readers.exchange_rates.from_usd_to_lcu(country_name, year)
        tau_sif = readers.oecd_econ.read_tau_sif(country_name, year)

        total_firm_deposits = (
            readers.eurostat.get_total_nonfin_firm_deposits(proxy_country, year) * exchange_rate_from_eur
            if proxy_country
            else readers.eurostat.get_total_nonfin_firm_deposits(country_name, year)
        )

        match firm_configuration.constructor:
            case "Default":
                firm_size_zetas = readers.oecd_econ.read_firm_size_zetas(
                    country_name,
                    year,
                )
                if firm_size_zetas is None:
                    firm_size_zetas = readers.ons.get_firm_size_zetas()

                keys = set(firm_size_zetas.keys())
                if len(keys) != len(industries):
                    avg_zeta = np.mean([firm_size_zetas[key] for key in keys])
                    for i in range(len(industries)):
                        firm_size_zetas[i] = avg_zeta

                firm_data = initialise_basic_firm_fields(
                    firm_data,
                    industry_data,
                    n_employees_per_industry,
                    n_firms_per_industry,
                    firm_size_zetas,
                    exchange_rate,
                    tau_sif,
                )
            case "Compustat":
                compustat_data = readers.compustat_firms.get_firm_data(country_name)
                firm_data = initialise_basic_firm_fields_compustat(
                    firm_data=firm_data,
                    compustat_data=compustat_data,
                    industry_data=industry_data,
                    n_employees_per_industry=n_employees_per_industry,
                    n_firms_per_industry=n_firms_per_industry,
                    exchange_rate=exchange_rate,
                    tau_sif=tau_sif,
                )

        total_firm_deposits = (
            readers.eurostat.get_total_nonfin_firm_deposits(country_name, year) * exchange_rate_from_eur
        )
        total_firm_debt = readers.eurostat.get_total_nonfin_firm_debt(country_name, year) * exchange_rate_from_eur

        capital_inputs_productivity_matrix = industry_data["capital_inputs_productivity_matrix"].values
        intermediate_inputs_productivity_matrix = industry_data["intermediate_inputs_productivity_matrix"].values
        capital_inputs_depreciation_matrix = industry_data["capital_inputs_depreciation_matrix"].values

        output = industry_data["industry_vectors"]["Output in USD"].values
        labour_productivity = output / n_employees_per_industry

        # TODO needs to be updated if function parameters change

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
            assume_zero_initial_debt=firm_configuration.zero_initial_debt,
            assume_zero_initial_deposits=firm_configuration.zero_initial_deposits,
            capital_inputs_utilisation_rate=firm_configuration.capital_inputs_utilisation_rate,
            initial_inventory_to_input_fraction=firm_configuration.initial_inventory_to_input_fraction,
            intermediate_inputs_utilisation_rate=firm_configuration.intermediate_inputs_utilisation_rate,
        )

        firm_data["Employees ID"] = [[] for _ in range(n_firms)]

        if emission_factors is not None:
            emitting_industries = ["B05a", "B05b", "B05c", "C19"]
            # get indices of emitting industries
            emitting_indices = [list(industries).index(industry) for industry in emitting_industries]
            emitting_intermediate_inputs = used_intermediate_inputs[:, emitting_indices]
            input_emissions = emitting_intermediate_inputs @ emission_factors.emissions_array
            firm_data["Input Emissions"] = input_emissions

            capital_emissions = used_capital_inputs[:, emitting_indices] @ emission_factors.emissions_array
            firm_data["Capital Emissions"] = capital_emissions

            # decompose emissions of oil, gas, coal and refined products emissions
            for i, name in enumerate(["Coal", "Gas", "Oil", "Refined Products"]):
                firm_data[f"{name} Input Emissions"] = (
                    used_intermediate_inputs[:, emitting_indices[i]] * emission_factors.emissions_array[i]
                )
                # same for capital emissions
                firm_data[f"{name} Capital Emissions"] = (
                    used_capital_inputs[:, emitting_indices[i]] * emission_factors.emissions_array[i]
                )

            firm_data.loc[
                firm_data["Industry"] == emitting_indices[-1],
                ["Input Emissions", "Capital Emissions"],
            ] = 0.0

            zero_columns = [
                "Oil Input Emissions",
                "Gas Input Emissions",
                "Coal Input Emissions",
                "Oil Capital Emissions",
                "Gas Capital Emissions",
                "Coal Capital Emissions",
            ]

            firm_data.loc[
                firm_data["Industry"] == emitting_indices[-1],
                zero_columns,
            ] = 0.0

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
            labour_productivity_by_industry=labour_productivity,
        )

    def reset_function_parameters(
        self,
        capital_inputs_utilisation_rate: float,
        initial_inventory_to_input_fraction: float,
        intermediate_inputs_utilisation_rate: float,
        zero_initial_debt: bool,
        zero_initial_deposits: bool,
    ):
        (
            capital_inputs_stock,
            intermediate_inputs_stock,
            used_capital_inputs,
            used_intermediate_inputs,
        ) = function_parameters_dependent_initialisation(
            firm_data=self.firm_data,
            intermediate_inputs_productivity_matrix=self.intermediate_inputs_productivity_matrix,
            capital_inputs_depreciation_matrix=self.capital_inputs_depreciation_matrix,
            capital_inputs_productivity_matrix=self.capital_inputs_productivity_matrix,
            total_firm_deposits=self.total_firm_deposits,
            total_firm_debt=self.total_firm_debt,
            assume_zero_initial_debt=zero_initial_debt,
            assume_zero_initial_deposits=zero_initial_deposits,
            capital_inputs_utilisation_rate=capital_inputs_utilisation_rate,
            initial_inventory_to_input_fraction=initial_inventory_to_input_fraction,
            intermediate_inputs_utilisation_rate=intermediate_inputs_utilisation_rate,
        )

        self.intermediate_inputs_stock = intermediate_inputs_stock
        self.used_intermediate_inputs = used_intermediate_inputs
        self.capital_inputs_stock = capital_inputs_stock
        self.used_capital_inputs = used_capital_inputs

    # THINGS TO BE SET AFTER MATCHING
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
        short_term_loan_interest: np.ndarray,
        long_term_loan_interest: np.ndarray,
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
        interest_on_loans = short_term_loan_interest.sum(axis=0) + long_term_loan_interest.sum(axis=0)

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

    def set_firm_debt_installments(
        self,
        long_term_installments: np.ndarray,
        short_term_installments: np.ndarray,
    ) -> None:
        debt_installments = long_term_installments.sum(axis=0) + short_term_installments.sum(axis=0)
        self.firm_data["Debt Installments"] = debt_installments

    def set_additional_initial_conditions(
        self,
        industry_data: dict[str, pd.DataFrame],
        synthetic_banks: SyntheticBanks,
        long_term_loans: LongtermLoans,
        short_term_loans: ShorttermLoans,
        tax_data: TaxData,
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
            short_term_loan_interest=short_term_loans.interest,
            long_term_loan_interest=long_term_loans.interest,
        )

        self.set_firm_profits(
            intermediate_inputs_productivity_matrix=industry_data["intermediate_inputs_productivity_matrix"].values,
            capital_inputs_depreciation_matrix=industry_data["capital_inputs_depreciation_matrix"].values,
            tau_sif=tax_data.employer_social_insurance_tax,
        )

        self.set_unit_costs(
            intermediate_inputs_productivity_matrix=industry_data["intermediate_inputs_productivity_matrix"].values,
            capital_inputs_depreciation_matrix=industry_data["capital_inputs_depreciation_matrix"].values,
            tau_sif=tax_data.employer_social_insurance_tax,
        )

        self.set_corporate_taxes_paid(
            tau_firm=tax_data.profit_tax,
        )

        self.set_firm_debt_installments(
            long_term_installments=long_term_loans.installments,
            short_term_installments=short_term_loans.installments,
        )
