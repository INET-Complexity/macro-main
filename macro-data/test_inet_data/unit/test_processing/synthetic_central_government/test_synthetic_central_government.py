import pathlib

import numpy as np
import pandas as pd

from inet_data.processing.synthetic_central_government.default_synthetic_central_government import (
    DefaultSyntheticCGovernment,
)

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


class TestSyntheticCentralGovernment:
    def test__create(self, readers):
        central_gov = DefaultSyntheticCGovernment.from_readers(readers=readers, country_name="FRA", year=2014)
        # Check if we have all the necessary fields
        for central_gov_field in [
            "Debt",
            "Total Unemployment Benefits",
            "Other Social Benefits",
        ]:
            assert central_gov_field in central_gov.central_gov_data.columns

        # Check if there are any missing values
        assert not np.any(pd.isna(central_gov.central_gov_data))
