import pathlib

import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.processing.synthetic_government_entities.default_synthetic_government_entities import (
    DefaultSyntheticGovernmentEntities,
)

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


class TestSyntheticGovernmentEntities:
    def test__create(self, readers, all_exogenous_data, industry_data):
        exogenous_country_data = all_exogenous_data.get("FRA", None)
        synth_gov_fra = DefaultSyntheticGovernmentEntities.from_readers(
            readers=readers,
            country_name=Country("FRA"),
            year=2014,
            exogenous_country_data=exogenous_country_data,
            industry_data=industry_data["FRA"],
            single_government_entity=True,
            quarter=1,
        )
        assert synth_gov_fra.number_of_entities == 1
