from abc import ABC, abstractmethod

import pandas as pd


class SyntheticCentralBank(ABC):
    @abstractmethod
    def __init__(
        self,
        country_name: str,
        year: int,
        central_bank_data: pd.DataFrame,
    ):
        self.country_name = country_name
        self.year = year

        # Bank data
        self.central_bank_data = central_bank_data
