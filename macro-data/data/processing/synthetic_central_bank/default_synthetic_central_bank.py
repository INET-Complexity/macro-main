import numpy as np

from data.processing.synthetic_central_bank.synthetic_central_bank import (
    SyntheticCentralBank,
)


class SyntheticDefaultCentralBanks(SyntheticCentralBank):
    def __init__(
        self,
        country_name: str,
        year: int,
    ):
        super().__init__(
            country_name,
            year,
        )

    def create(self, initial_policy_rate: float) -> None:
        self.create_agents(initial_policy_rate=initial_policy_rate)
