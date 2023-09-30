import pathlib
import numpy as np
import pandas as pd

from inet_data.processing.synthetic_government_entities.default_synthetic_government_entities import (
    SyntheticDefaultGovernmentEntities,
)

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


class TestSyntheticGovernmentEntities:
    def test__create(self, readers):
        gov_entities = SyntheticDefaultGovernmentEntities(
            country_name="FRA",
            year=2014,
        )
        gov_entities.create(
            single_government_entity=False,
            monthly_govt_consumption_in_lcu=np.full(18, 3.0),
            monthly_govt_consumption_in_usd=np.full(18, 3.5),
            total_monthly_value_added_in_lcu=100.0,
            total_number_of_firms=18,
            total_gov_consumption_growth=np.array([0.04, 0.03]),
        )

        # Check if we have all the necessary fields
        for gov_entities_field in ["Consumption in LCU", "Consumption in USD"]:
            assert gov_entities_field in gov_entities.gov_entity_data.columns

        # Check if there are any missing values
        assert not np.any(pd.isna(gov_entities.gov_entity_data))
