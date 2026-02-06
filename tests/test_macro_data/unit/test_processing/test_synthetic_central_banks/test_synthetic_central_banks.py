import pathlib

import numpy as np
import pandas as pd

from macro_data.configuration.dataconfiguration import CentralBankDataConfiguration
from macro_data.processing.synthetic_central_bank.default_synthetic_central_bank import (
    DefaultSyntheticCentralBank,
)

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


class TestSyntheticCentralBanks:
    def test__create(self, readers, exogenous_data):
        central_bank_config = CentralBankDataConfiguration()

        central_banks = DefaultSyntheticCentralBank.from_readers(
            country_name="FRA",
            year=2014,
            readers=readers,
            quarter=1,
            central_bank_configuration=central_bank_config,
            exogenous_data=exogenous_data,
        )

        # Check if we have all the necessary fields
        for central_bank_field in ["policy_rate"]:
            assert central_bank_field in central_banks.central_bank_data.columns

        # Check if there are any missing values
        assert not np.any(pd.isna(central_banks.central_bank_data))
