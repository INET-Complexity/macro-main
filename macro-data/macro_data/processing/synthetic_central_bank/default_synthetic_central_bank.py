import pandas as pd

from macro_data.processing.synthetic_central_bank.synthetic_central_bank import (
    SyntheticCentralBank,
)
from macro_data.readers.default_readers import DataReaders


class DefaultSyntheticCentralBank(SyntheticCentralBank):
    """
    A class representing synthetic central banks.

    The central bank data is stored in a pandas DataFrame with a single column:
        - Policy Rate: The policy rate.

    Attributes:
        country_name (str): The name of the country.
        year (int): The year.
        central_bank_data (pd.DataFrame): The central bank data, containing the policy rate and possibly more data.

    Methods:
        __init__(country_name, year, central_bank_data): Initializes a SyntheticDefaultCentralBanks instance.
        from_readers(country_name, year, readers): Initializes a SyntheticDefaultCentralBanks instance from readers.
    """

    def __init__(
        self,
        country_name: str,
        year: int,
        central_bank_data: pd.DataFrame,
    ):
        super().__init__(
            country_name,
            year,
            central_bank_data,
        )

    @classmethod
    def from_readers(cls, country_name: str, year: int, readers: DataReaders):
        """
        Initializes a SyntheticCentralBank object using data from DataReaders,
        in particular storing the central bank policy rate.

        Args:
            country_name (str): The name of the country.
            year (int): The year.
            readers (DataReaders): An instance of DataReaders.

        Returns:
            SyntheticCentralBank: An instance of SyntheticCentralBank.
        """
        initial_policy_rate = readers.policy_rates.cb_policy_rate(country_name, year)
        central_bank_data = pd.DataFrame({"Policy Rate": [initial_policy_rate]})
        return cls(country_name, year, central_bank_data)
