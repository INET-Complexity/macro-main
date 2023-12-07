import pathlib

import numpy as np
import pandas as pd

from inet_data.processing.synthetic_central_bank.default_synthetic_central_bank import (
    SyntheticDefaultCentralBanks,
)

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


class TestSyntheticCentralBanks:
    def test__create(self):
        central_banks = SyntheticDefaultCentralBanks(
            country_name="FRA",
            year=2014,
        )
        central_banks.create(initial_policy_rate=0.02)

        # Check if we have all the necessary fields
        for central_bank_field in ["Policy Rate"]:
            assert central_bank_field in central_banks.central_bank_data.columns

        # Check if there are any missing values
        assert not np.any(pd.isna(central_banks.central_bank_data))
