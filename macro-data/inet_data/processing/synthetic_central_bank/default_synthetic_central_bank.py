import numpy as np
import pandas as pd

from inet_data.processing.synthetic_central_bank.synthetic_central_bank import (
    SyntheticCentralBank,
)
from inet_data.readers.default_readers import DataReaders


class SyntheticDefaultCentralBanks(SyntheticCentralBank):
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
    def init_from_readers(cls, country_name: str, year: int, readers: DataReaders):
        initial_policy_rate = readers.policy_rates.cb_policy_rate(country_name, year)
        central_bank_data = pd.DataFrame({"Policy Rate": [initial_policy_rate]})
        return cls(country_name, year, central_bank_data)
