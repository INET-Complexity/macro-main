import pathlib

import numpy as np
import pandas as pd

from inet_data.processing.synthetic_rest_of_the_world.default_synthetic_rest_of_the_world import (
    DefaultSyntheticRestOfTheWorld,
)

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


class TestSyntheticRestOfTheWorld:
    def test__create(self, readers, all_exogenous_data, industry_data):
        rest_of_the_world = DefaultSyntheticRestOfTheWorld.init_from_readers(
            year=2014,
            readers=readers,
            exogenous_row_data=all_exogenous_data.get("ROW", None),
            row_industry_data=industry_data,
        )
        # Check if we have all the necessary fields
        for row_field in [
            "Imports in USD",
            "Imports in LCU",
            "Exports",
            "Price in USD",
            "Price in LCU",
        ]:
            assert row_field in rest_of_the_world.row_data.columns

        # Check if there are any missing values
        assert not np.any(pd.isna(rest_of_the_world.row_data))
