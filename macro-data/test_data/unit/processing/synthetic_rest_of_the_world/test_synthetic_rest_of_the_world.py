import pathlib
import numpy as np
import pandas as pd

from data.processing.synthetic_rest_of_the_world.default_synthetic_rest_of_the_world import (
    DefaultSyntheticRestOfTheWorld,
)

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


class TestSyntheticRestOfTheWorld:
    def test__create(self, industry_data):
        rest_of_the_world = DefaultSyntheticRestOfTheWorld(year=2014)
        rest_of_the_world.create(
            row_exports=industry_data["ROW"]["industry_vectors"]["Exports in USD"],
            row_imports=industry_data["ROW"]["industry_vectors"]["Imports in USD"],
            exchange_rate_usd_to_lcu=1.0,
            row_exports_data_growth=np.array([0.03, 0.04]),
            row_imports_data_growth=np.array([0.02, 0.03]),
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
