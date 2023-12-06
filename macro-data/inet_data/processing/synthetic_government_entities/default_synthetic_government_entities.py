from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression

from inet_data.processing.synthetic_government_entities.synthetic_government_entities import (
    SyntheticGovernmentEntities,
)


class SyntheticDefaultGovernmentEntities(SyntheticGovernmentEntities):
    def __init__(
        self,
        country_name: str,
        year: int,
    ):
        super().__init__(
            country_name,
            year,
        )

    def set_gov_entity_total_consumption(
        self,
        monthly_govt_consumption_in_usd: np.ndarray,
        monthly_govt_consumption_in_lcu: np.ndarray,
        total_gov_consumption_growth: Optional[np.ndarray],
    ) -> None:
        self.gov_entity_data["Consumption in LCU"] = monthly_govt_consumption_in_lcu
        self.gov_entity_data["Consumption in USD"] = monthly_govt_consumption_in_usd
        if total_gov_consumption_growth is None or len(total_gov_consumption_growth) == 0:
            self.government_consumption_model = None
        else:
            self.government_consumption_model = LinearRegression().fit(
                [[0], [1]], [np.nanmean(total_gov_consumption_growth), np.nanmean(total_gov_consumption_growth)]
            )
