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
    ):
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
        """
        Initialize a SyntheticBanks object from data readers.
        This method creates a single bank or multiple banks, depending on the single_bank
        flag and on the number of bank branches
        in the country obtained from the data.

        Bank equity is set to the total bank equity (obtained from Eurostat) in the country
        divided by the number of banks.


        Args:
            cls (class): The class object.
            single_bank (bool): Flag indicating whether to create a single bank or multiple banks.
            country_name (Country): The name of the country.
            year (int): The year.
            quarter (int): The quarter.
            readers (DataReaders): The data readers object.
            scale (int): The scaling factor.
            banks_data_configuration (BanksDataConfiguration): The banks data configuration object.
            exchange_rate_from_eur (float): The exchange rate from EUR to the local currency.
            proxy_eu_country (Optional[Country]): The proxy EU country.
            inflation_data (pd.DataFrame): The inflation data.

        Returns:
            SyntheticBanks: The initialized SyntheticBanks object.
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

        bank_equity = readers.eurostat.get_total_bank_equity(country=country_name, year=year) * exchange_rate_from_eur
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
        """
        Initialize a SyntheticBanks object from Compustat data readers.
        This method creates a single bank or multiple banks, depending on the number of banks
        in the country obtained from the data.

        Bank equity is set to the total bank equity (obtained from Compustat) in the country
        divided by the number of banks.

        Args:
            cls (class): The class object.
            country_name (Country): The name of the country.
            year (int): The year.
            readers (DataReaders): The data readers object.
            single_bank (bool): Flag indicating whether to create a single bank or multiple banks.
            scale (int): The scaling factor.
            quarter (int): The quarter.
            inflation_data (pd.DataFrame): The inflation data.
            exchange_rate_from_eur (float): The exchange rate from EUR to the local currency.
            proxy_eu_country (Optional[Country]): The proxy EU country.

        Returns:
            SyntheticBanks: The initialized SyntheticBanks object.
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

        total_bank_equity = readers.eurostat.get_total_bank_equity(country=country_name, year=year)

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

    @classmethod
    def initialise_rates(cls, country_name, inflation_data, proxy_eu_country, quarter, readers, year):
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
        self.bank_data["Equity"] = np.full(self.number_of_banks, bank_equity / self.number_of_banks)

    def set_deposits_from_firms(self, firm_deposits: np.ndarray) -> None:
        initial_deposits_from_firms = np.zeros(self.number_of_banks)
        for bank_id in range(self.number_of_banks):
            corr_firms = np.array(self.bank_data["Corresponding Firms ID"][bank_id])
            initial_deposits_from_firms[bank_id] += firm_deposits[corr_firms].sum()
        self.bank_data["Deposits from Firms"] = initial_deposits_from_firms

    def set_deposits_from_households(self, household_deposits: np.ndarray) -> None:
        initial_deposits_from_households = np.zeros(self.number_of_banks)
        for bank_id in range(self.number_of_banks):
            corr_households = np.array(self.bank_data["Corresponding Households ID"][bank_id])
            initial_deposits_from_households[bank_id] += household_deposits[corr_households].sum()
        self.bank_data["Deposits from Households"] = initial_deposits_from_households

    def set_loans_to_firms(self, firm_debt: np.ndarray) -> None:
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
        initial_mortgages_to_households = np.zeros(self.number_of_banks)
        initial_other_loans_to_households = np.zeros(self.number_of_banks)
        for bank_id in range(self.number_of_banks):
            corr_firms = np.array(self.bank_data["Corresponding Households ID"][bank_id])
            initial_mortgages_to_households[bank_id] += household_mortgage_debt[corr_firms].sum()
            initial_other_loans_to_households[bank_id] += household_other_debt[corr_firms].sum()
        self.bank_data["Mortgages to Households"] = initial_mortgages_to_households
        self.bank_data["Consumption Loans to Households"] = initial_other_loans_to_households

    def set_bank_deposits(
        self,
        firm_deposits: np.ndarray,
        firm_debt: np.ndarray,
        household_deposits: np.ndarray,
        household_debt: np.ndarray,
    ) -> None:
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
