from abc import ABC, abstractmethod

import pandas as pd


class SyntheticCentralBank(ABC):
    @abstractmethod
    def __init__(
        self,
        country_name: str,
        year: int,
    ):
        self.country_name = country_name
        self.year = year

        # Bank data
        self.central_bank_data = pd.DataFrame()

    @abstractmethod
    def create(self, initial_policy_rate: float) -> None:
        pass

    def create_agents(self, initial_policy_rate: float) -> None:
        self.set_central_bank_policy_rate(initial_policy_rate)

    def set_central_bank_policy_rate(self, initial_policy_rate: float) -> None:
        self.central_bank_data["Policy Rate"] = [initial_policy_rate]
