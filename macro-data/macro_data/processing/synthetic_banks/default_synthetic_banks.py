import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.configuration.dataconfiguration import BanksDataConfiguration
from macro_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks

from macro_data.readers.default_readers import DataReaders


class DefaultSyntheticBanks(SyntheticBanks):
    def __init__(
        self,
        country_name: str,
        year: int,
        number_of_banks: int,
        bank_data: pd.DataFrame,
    ):
        super().__init__(
            country_name,
            year,
            number_of_banks,
            bank_data,
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
        exchange_rate_from_eur: float = 1.0,
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
            readers (DataReaders): The data readers object.
            scale (int): The scaling factor.
            banks_data_configuration (BanksDataConfiguration): The banks data configuration object.
            exchange_rate_from_eur (float): The exchange rate from EUR to the local currency.

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

        return cls(country_name, year, number_of_banks, bank_data)

    @classmethod
    def from_readers_compustat(
        cls, country_name: Country, year: int, readers: DataReaders, single_bank: bool, scale: int
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

        # temp, reseed
        np.random.seed(0)
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

        return cls(country_name, year, number_of_banks, bank_data)

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
