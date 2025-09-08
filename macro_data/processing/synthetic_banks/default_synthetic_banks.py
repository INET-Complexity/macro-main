"""Module for preprocessing synthetic banking system data.

This module provides a concrete implementation for preprocessing banking system data
that will be used to initialize behavioral models. Key preprocessing includes:

1. Data Collection and Processing:
   - Bank balance sheet data preparation
   - Historical rate parameter estimation
   - Initial state calculations

2. Bank Data Organization:
   - Standard bank data preprocessing
   - Compustat-based data preprocessing
   - Data validation and consistency checks

3. Parameter Estimation:
   - Interest rate parameters
   - Balance sheet ratios
   - Market share calculations

Note:
    This module is NOT used for simulating bank behavior. It preprocesses
    data that will be used to initialize behavioral models in the simulation package.
    The actual banking decisions and operations are implemented elsewhere.
"""

from typing import Optional

import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.configuration.dataconfiguration import BanksDataConfiguration
from macro_data.processing.synthetic_banks.rates_utils import (
    default_rate_values,
    fit_firm_models,
    fit_household_models,
    fit_mortgage_models,
    rates_dataframe,
)
from macro_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks
from macro_data.readers.default_readers import DataReaders


class DefaultSyntheticBanks(SyntheticBanks):
    """Default implementation for preprocessing banking system data.

    This class preprocesses and organizes banking system data by collecting historical
    data and estimating parameters. These will be used to initialize behavioral models,
    but this class does NOT implement any behavioral logic.

    The class provides two preprocessing paths:
    1. Standard preprocessing using OECD/Eurostat data
    2. Compustat-based preprocessing for detailed bank profiles

    The preprocessing includes:
    - Bank balance sheet data organization
    - Interest rate parameter estimation
    - Relationship mapping (bank-firm, bank-household)
    - Initial state calculations

    Note:
        This is a data container class. The actual banking behavior (lending,
        rate setting, etc.) is implemented in the simulation package, which uses
        these preprocessed parameters.

    Attributes:
        country_name (str): Country identifier for data collection
        year (int): Reference year for preprocessing
        number_of_banks (int): Number of banks to preprocess data for
        bank_data (pd.DataFrame): Preprocessed bank-level data
        quarter (int): Reference quarter for preprocessing
        firm_passthrough (float): Estimated rate adjustment parameter
        firm_ect (float): Estimated error correction parameter
        firm_rate (float): Initial firm loan rate
        hh_consumption_passthrough (float): Estimated consumer rate parameter
        hh_consumption_ect (float): Estimated consumer ECT parameter
        hh_consumption_rate (float): Initial consumer loan rate
        hh_mortgage_passthrough (float): Estimated mortgage rate parameter
        hh_mortgage_ect (float): Estimated mortgage ECT parameter
        hh_mortgage_rate (float): Initial mortgage rate
        proxy_country (Optional[Country]): Country to use as proxy when missing data
    """

    def __init__(
        self,
        country_name: str,
        year: int,
        number_of_banks: int,
        bank_data: pd.DataFrame,
        quarter: int,
        firm_passthrough: float,
        firm_ect: float,
        firm_rate: float,
        hh_consumption_passthrough: float,
        hh_consumption_ect: float,
        hh_consumption_rate: float,
        hh_mortgage_passthrough: float,
        hh_mortgage_ect: float,
        hh_mortgage_rate: float,
        proxy_country: Optional[Country] = None,
    ):
        """Initialize the banking system data container.

        Args:
            country_name (str): Country identifier for data collection
            year (int): Reference year for preprocessing
            number_of_banks (int): Number of banks to preprocess data for
            bank_data (pd.DataFrame): Initial data to preprocess
            quarter (int): Reference quarter for preprocessing
            firm_passthrough (float): Estimated rate adjustment parameter
            firm_ect (float): Estimated error correction parameter
            firm_rate (float): Initial firm loan rate
            hh_consumption_passthrough (float): Estimated consumer rate parameter
            hh_consumption_ect (float): Estimated consumer ECT parameter
            hh_consumption_rate (float): Initial consumer loan rate
            hh_mortgage_passthrough (float): Estimated mortgage rate parameter
            hh_mortgage_ect (float): Estimated mortgage ECT parameter
            hh_mortgage_rate (float): Initial mortgage rate
            proxy_country (Optional[Country]): Country to use as proxy when missing data
        """
        super().__init__(
            country_name,
            year,
            number_of_banks,
            bank_data,
            quarter=quarter,
            firm_passthrough=firm_passthrough,
            firm_ect=firm_ect,
            firm_rate=firm_rate,
            hh_consumption_passthrough=hh_consumption_passthrough,
            hh_consumption_ect=hh_consumption_ect,
            hh_consumption_rate=hh_consumption_rate,
            hh_mortgage_passthrough=hh_mortgage_passthrough,
            hh_mortgage_ect=hh_mortgage_ect,
            hh_mortgage_rate=hh_mortgage_rate,
        )
        self.proxy_country = proxy_country

    @classmethod
    def from_readers(
        cls,
        single_bank: bool,
        country_name: Country,
        year: int,
        readers: DataReaders,
        scale: int,
        banks_data_configuration: BanksDataConfiguration,
        quarter: int,
        inflation_data: pd.DataFrame,
        exchange_rate_from_eur: float = 1.0,
        proxy_eu_country: Optional[Country] = None,
    ) -> "DefaultSyntheticBanks":
        """Create a preprocessed banking system data container using standard data sources.

        This method preprocesses data using OECD/Eurostat sources to prepare:
        1. Bank balance sheet data (scaled appropriately)
        2. Initial rate parameters
        3. Bank relationship mappings

        For standard preprocessing:
        - Number of banks is derived from actual bank branches (scaled)
        - Bank equity is distributed based on historical data
        - Rate parameters are estimated from historical rates

        Args:
            single_bank (bool): Whether to preprocess data for a single bank
            country_name (Country): Country to preprocess data for
            year (int): Reference year for preprocessing
            readers (DataReaders): Data source readers
            scale (int): Scaling factor for bank numbers
            banks_data_configuration (BanksDataConfiguration): Preprocessing configuration
            quarter (int): Reference quarter for preprocessing
            inflation_data (pd.DataFrame): Historical inflation data
            exchange_rate_from_eur (float, optional): Exchange rate for conversion. Defaults to 1.0.
            proxy_eu_country (Optional[Country], optional): EU country for proxy data. Defaults to None.

        Returns:
            DefaultSyntheticBanks: Container with preprocessed banking system data
        """
        if banks_data_configuration.constructor == "Compustat":
            return cls.from_readers_compustat(
                country_name=country_name,
                year=year,
                readers=readers,
                single_bank=single_bank,
                scale=scale,
                quarter=quarter,
                inflation_data=inflation_data,
                exchange_rate_from_eur=exchange_rate_from_eur,
                proxy_eu_country=proxy_eu_country,
            )

        if single_bank:
            number_of_banks = 1
        else:
            bank_branches = readers.oecd_econ.read_number_of_banks(country=country_name, year=year)
            number_of_banks = max(1, int(bank_branches / scale))

        bank_equity = (
            readers.eurostat.get_total_bank_equity(country=country_name, year=year, proxy_country=proxy_eu_country)
            * exchange_rate_from_eur
        )
        bank_data = pd.DataFrame({"Equity": np.ones(number_of_banks) * bank_equity / number_of_banks})
        bank_data["Deposits from Firms"] = np.ones(number_of_banks)
        bank_data["Deposits from Households"] = np.ones(number_of_banks)
        bank_data["Loans to Firms"] = np.ones(number_of_banks)
        bank_data["Consumption Loans to Households"] = np.ones(number_of_banks)
        bank_data["Mortgages to Households"] = np.ones(number_of_banks)
        bank_data["Loans to Households"] = (
            bank_data["Consumption Loans to Households"] + bank_data["Mortgages to Households"]
        )

        (
            firm_ect,
            firm_passthrough,
            firm_rate,
            hh_consumption_ect,
            hh_consumption_passthrough,
            hh_consumption_rate,
            hh_mortgage_ect,
            hh_mortgage_passthrough,
            hh_mortgage_rate,
        ) = cls.initialise_rates(country_name, inflation_data, proxy_eu_country, quarter, readers, year)

        initial_central_bank_policy_rate = (
            readers.policy_rates.get_policy_rates(country_name).loc[f"{year}-Q{quarter}", "Policy Rate"].values[0]
        )

        bank_data["Interest Rates on Firm Deposits"] = initial_central_bank_policy_rate
        bank_data["Interest Rates on Household Deposits"] = initial_central_bank_policy_rate

        return cls(
            country_name,
            year,
            number_of_banks,
            bank_data,
            firm_passthrough=firm_passthrough,
            firm_ect=firm_ect,
            firm_rate=firm_rate,
            hh_consumption_passthrough=hh_consumption_passthrough,
            hh_consumption_ect=hh_consumption_ect,
            hh_consumption_rate=hh_consumption_rate,
            hh_mortgage_passthrough=hh_mortgage_passthrough,
            hh_mortgage_ect=hh_mortgage_ect,
            hh_mortgage_rate=hh_mortgage_rate,
            quarter=quarter,
            proxy_country=proxy_eu_country,
        )

    @classmethod
    def from_readers_compustat(
        cls,
        country_name: Country,
        year: int,
        readers: DataReaders,
        single_bank: bool,
        scale: int,
        quarter: int,
        inflation_data: pd.DataFrame,
        exchange_rate_from_eur: float = 1.0,
        proxy_eu_country: Optional[Country] = None,
    ) -> "DefaultSyntheticBanks":
        """Create a preprocessed banking system data container using Compustat data.

        This method preprocesses detailed Compustat data to prepare:
        1. Historical balance sheet information
        2. Actual deposit and loan distributions
        3. Historical equity levels and ratios

        The preprocessing steps:
        1. Fetch and filter relevant Compustat bank data
        2. Sample and scale data appropriately
        3. Align with country-level totals
        4. Estimate initial parameters

        Args:
            country_name (Country): Country to preprocess data for
            year (int): Reference year for preprocessing
            readers (DataReaders): Data source readers
            single_bank (bool): Whether to preprocess data for a single bank
            scale (int): Scaling factor for bank numbers
            quarter (int): Reference quarter for preprocessing
            inflation_data (pd.DataFrame): Historical inflation data
            exchange_rate_from_eur (float, optional): Exchange rate for conversion. Defaults to 1.0.
            proxy_eu_country (Optional[Country], optional): EU country for proxy data. Defaults to None.

        Returns:
            DefaultSyntheticBanks: Container with preprocessed banking system data
        """
        compustat_data = readers.compustat_banks.get_country_data(
            country=country_name, exchange_rate=readers.exchange_rates.from_usd_to_lcu(country_name, year)
        )

        # select only positive debt
        compustat_data = compustat_data[compustat_data["Debt"] > 0]

        if single_bank:
            number_of_banks = 1
        else:
            oecd_banks = readers.oecd_econ.read_number_of_banks(country=country_name, year=year)
            number_of_banks = max(1, int(oecd_banks / scale))

        banks_inds = np.random.choice(range(len(compustat_data)), number_of_banks, replace=True)

        compustat_selection = compustat_data.iloc[banks_inds]

        total_bank_equity = readers.eurostat.get_total_bank_equity(
            country=country_name, year=year, proxy_country=proxy_eu_country
        )

        bank_data = pd.DataFrame(
            {
                "Deposits from Firms": compustat_selection["Deposits"].values,
                "Deposits from Households": compustat_selection["Deposits"].values,
                "Loans to Firms": compustat_selection["Debt"].values,
                "Consumption Loans to Households": compustat_selection["Debt"].values,
                "Mortgages to Households": compustat_selection["Debt"].values,
                "Loans to Households": compustat_selection["Debt"].values,
                "Equity": compustat_selection["Equity"].values,
            }
        )
        bank_data["Equity"] *= total_bank_equity / bank_data["Equity"].sum()

        (
            firm_ect,
            firm_passthrough,
            firm_rate,
            hh_consumption_ect,
            hh_consumption_passthrough,
            hh_consumption_rate,
            hh_mortgage_ect,
            hh_mortgage_passthrough,
            hh_mortgage_rate,
        ) = cls.initialise_rates(country_name, inflation_data, proxy_eu_country, quarter, readers, year)

        initial_central_bank_policy_rate = (
            readers.policy_rates.get_policy_rates(country_name).loc[f"{year}-Q{quarter}", "Policy Rate"].values[0]
        )

        bank_data["Interest Rates on Firm Deposits"] = initial_central_bank_policy_rate
        bank_data["Interest Rates on Household Deposits"] = initial_central_bank_policy_rate

        return cls(
            country_name,
            year,
            number_of_banks,
            bank_data,
            firm_passthrough=firm_passthrough,
            firm_ect=firm_ect,
            firm_rate=firm_rate,
            hh_consumption_passthrough=hh_consumption_passthrough,
            hh_consumption_ect=hh_consumption_ect,
            hh_consumption_rate=hh_consumption_rate,
            hh_mortgage_passthrough=hh_mortgage_passthrough,
            hh_mortgage_ect=hh_mortgage_ect,
            hh_mortgage_rate=hh_mortgage_rate,
            quarter=quarter,
            proxy_country=proxy_eu_country,
        )

    @classmethod
    def initialise_rates(cls, country_name, inflation_data, proxy_eu_country, quarter, readers, year):
        """Preprocess and estimate initial interest rate parameters.

        This method:
        1. Collects historical rate data
        2. Estimates rate adjustment parameters
        3. Calculates initial rates for different products

        The preprocessing includes parameter estimation for:
        - Firm loan rates
        - Consumer loan rates
        - Mortgage rates
        - Default parameters when data is insufficient

        Args:
            country_name (Country): Country to process data for
            inflation_data (pd.DataFrame): Historical inflation data
            proxy_eu_country (Optional[Country]): EU country for proxy data
            quarter (int): Reference quarter
            readers (DataReaders): Data source readers
            year (int): Reference year

        Returns:
            tuple: Nine estimated parameters:
                - firm_ect: Estimated error correction for firm rates
                - firm_passthrough: Estimated adjustment for firm rates
                - firm_rate: Initial firm rate
                - hh_consumption_ect: Estimated consumer rate ECT
                - hh_consumption_passthrough: Estimated consumer rate adjustment
                - hh_consumption_rate: Initial consumer rate
                - hh_mortgage_ect: Estimated mortgage ECT
                - hh_mortgage_passthrough: Estimated mortgage adjustment
                - hh_mortgage_rate: Initial mortgage rate
        """
        if country_name.is_eu_country:
            data_country = country_name
        else:
            if proxy_eu_country is None:
                raise ValueError("Proxy EU country is required for non-EU countries.")
            data_country = proxy_eu_country
        firm_rate = readers.ecb_reader.get_firm_rates(data_country)
        household_consumption_rate = readers.ecb_reader.get_household_consumption_rates(data_country)
        household_mortgage_rates = readers.ecb_reader.get_household_mortgage_rates(data_country)
        policy_rates = readers.policy_rates.get_policy_rates(data_country)
        npl_rates = readers.world_bank.get_npl_ratios(data_country)
        if any(
            [
                firm_rate is None,
                household_consumption_rate is None,
                household_mortgage_rates is None,
                npl_rates is None,
            ]
        ):
            firm_passthrough, firm_ect, firm_rate = default_rate_values(policy_rates)
            hh_consumption_passthrough, household_consumption_ect, household_consumption_rate = default_rate_values(
                policy_rates
            )
            hh_mortgage_passthrough, household_mortgages_ect, household_mortgages_rate = default_rate_values(
                policy_rates
            )
        else:
            df = rates_dataframe(
                firm_rate,
                household_consumption_rate,
                household_mortgage_rates,
                inflation_data,
                npl_rates,
                policy_rates,
                year,
                quarter,
            )

            firm_passthrough, firm_ect, firm_rate = fit_firm_models(df, n_lags=1)
            hh_consumption_passthrough, household_consumption_ect, household_consumption_rate = fit_household_models(
                df, n_lags=1
            )
            hh_mortgage_passthrough, household_mortgages_ect, household_mortgages_rate = fit_mortgage_models(
                df, n_lags=1
            )
        return (
            firm_ect,
            firm_passthrough,
            firm_rate,
            household_consumption_ect,
            hh_consumption_passthrough,
            household_consumption_rate,
            household_mortgages_ect,
            hh_mortgage_passthrough,
            household_mortgages_rate,
        )

    def set_bank_equity(self, bank_equity: float) -> None:
        """Set the equity level for each bank.

        Args:
            bank_equity (float): Equity amount to set for each bank
        """
        self.bank_data["Equity"] = np.full(self.number_of_banks, bank_equity / self.number_of_banks)

    def set_deposits_from_firms(self, firm_deposits: np.ndarray) -> None:
        """Set the initial deposits from firms for each bank.

        Args:
            firm_deposits (np.ndarray): Array of deposit amounts by firm
        """
        initial_deposits_from_firms = np.zeros(self.number_of_banks)
        for bank_id in range(self.number_of_banks):
            corr_firms = np.array(self.bank_data["Corresponding Firms ID"][bank_id])
            initial_deposits_from_firms[bank_id] += firm_deposits[corr_firms].sum()
        self.bank_data["Deposits from Firms"] = initial_deposits_from_firms

    def set_deposits_from_households(self, household_deposits: np.ndarray) -> None:
        """Set the initial deposits from households for each bank.

        Args:
            household_deposits (np.ndarray): Array of deposit amounts by household
        """
        initial_deposits_from_households = np.zeros(self.number_of_banks)
        for bank_id in range(self.number_of_banks):
            corr_households = np.array(self.bank_data["Corresponding Households ID"][bank_id])
            initial_deposits_from_households[bank_id] += household_deposits[corr_households].sum()
        self.bank_data["Deposits from Households"] = initial_deposits_from_households

    def set_loans_to_firms(self, firm_debt: np.ndarray) -> None:
        """Set the initial loans to firms for each bank.

        Args:
            firm_debt (np.ndarray): Array of debt amounts by firm
        """
        initial_loans_to_firms = np.zeros(self.number_of_banks)
        for bank_id in range(self.number_of_banks):
            corr_firms = np.array(self.bank_data["Corresponding Firms ID"][bank_id])
            initial_loans_to_firms[bank_id] += firm_debt[corr_firms].sum()
        self.bank_data["Loans to Firms"] = initial_loans_to_firms

    def set_loans_to_households(
        self,
        household_mortgage_debt: np.ndarray,
        household_other_debt: np.ndarray,
    ) -> None:
        """Set the initial loans to households for each bank.

        Args:
            household_mortgage_debt (np.ndarray): Array of mortgage debt by household
            household_other_debt (np.ndarray): Array of non-mortgage debt by household
        """
        initial_mortgages_to_households = np.zeros(self.number_of_banks)
        initial_other_loans_to_households = np.zeros(self.number_of_banks)
        for bank_id in range(self.number_of_banks):
            corr_firms = np.array(self.bank_data["Corresponding Households ID"][bank_id])
            initial_mortgages_to_households[bank_id] += household_mortgage_debt[corr_firms].sum()
            initial_other_loans_to_households[bank_id] += household_other_debt[corr_firms].sum()
        self.bank_data["Mortgages to Households"] = initial_mortgages_to_households
        self.bank_data["Consumption Loans to Households"] = initial_other_loans_to_households
        self.bank_data["Loans to Households"] = initial_mortgages_to_households + initial_other_loans_to_households

    def set_bank_deposits(
        self,
        firm_deposits: np.ndarray,
        firm_debt: np.ndarray,
        household_deposits: np.ndarray,
        household_debt: np.ndarray,
    ) -> None:
        """Set the total deposits for each bank.

        Args:
            firm_deposits (np.ndarray): Array of firm deposit amounts
            firm_debt (np.ndarray): Array of firm debt amounts
            household_deposits (np.ndarray): Array of household deposit amounts
            household_debt (np.ndarray): Array of household debt amounts
        """
        initial_bank_deposits = np.zeros(self.number_of_banks)
        for bank_id in range(self.number_of_banks):
            corr_firms = np.array(self.bank_data["Corresponding Firms ID"][bank_id])
            corr_households = np.array(self.bank_data["Corresponding Households ID"][bank_id])
            total_firm_deposits = firm_deposits[corr_firms].sum()
            total_firm_debts = firm_debt[corr_firms].sum()
            total_household_deposits = household_deposits[corr_households].sum()
            total_household_debt = household_debt[corr_households].sum()

            # Compute initial deposits at the central bank
            initial_bank_deposits[bank_id] = (
                total_firm_deposits
                + total_household_deposits
                + self.bank_data["Equity"].values[bank_id]
                - total_firm_debts
                - total_household_debt
            )
        self.bank_data["Deposits"] = initial_bank_deposits
