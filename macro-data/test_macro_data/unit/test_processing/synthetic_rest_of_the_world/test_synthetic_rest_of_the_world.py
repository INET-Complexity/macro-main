import pathlib

import numpy as np
import pandas as pd

from macro_data.configuration.dataconfiguration import ROWDataConfiguration
from macro_data.processing.synthetic_rest_of_the_world.default_synthetic_rest_of_the_world import (
    DefaultSyntheticRestOfTheWorld,
)

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


class TestSyntheticRestOfTheWorld:
    def test__create(self, readers, all_exogenous_data, industry_data):
        row_data_config = ROWDataConfiguration()
        rest_of_the_world = DefaultSyntheticRestOfTheWorld.from_readers(
            year=2014,
            readers=readers,
            industry_data=industry_data,
            row_configuration=row_data_config,
            n_buyers=10,
            n_sellers_by_industry=np.ones(18),
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
