import numpy as np
import pandas as pd

from abc import abstractmethod, ABC

from typing import Optional


class SyntheticGovernmentEntities(ABC):
    @abstractmethod
    def __init__(
        self,
        country_name: str,
        year: int,
    ):
        self.country_name = country_name
        self.year = year

        # Government entity inet_data
        self.number_of_entities = None
        self.gov_entity_data = pd.DataFrame()

        # Consumption
        self.government_consumption_model = None

    def create(
        self,
        single_government_entity: bool,
        monthly_govt_consumption_in_usd: np.ndarray,
        monthly_govt_consumption_in_lcu: np.ndarray,
        total_monthly_value_added_in_lcu: float,
        total_number_of_firms: int,
        total_gov_consumption_growth: Optional[np.ndarray],
    ) -> None:
        self.set_number_of_entities(
            single_government_entity=single_government_entity,
            monthly_govt_consumption_in_lcu=monthly_govt_consumption_in_lcu,
            total_monthly_value_added_in_lcu=total_monthly_value_added_in_lcu,
            total_number_of_firms=total_number_of_firms,
        )
        self.set_gov_entity_total_consumption(
            monthly_govt_consumption_in_lcu=monthly_govt_consumption_in_lcu,
            monthly_govt_consumption_in_usd=monthly_govt_consumption_in_usd,
            total_gov_consumption_growth=total_gov_consumption_growth,
        )

    def set_number_of_entities(
        self,
        single_government_entity: bool,
        monthly_govt_consumption_in_lcu: np.ndarray,
        total_monthly_value_added_in_lcu: float,
        total_number_of_firms: int,
    ) -> None:
        if single_government_entity:
            self.number_of_entities = 1.0
        else:
            self.number_of_entities = int(
                max(
                    1,
                    total_number_of_firms * monthly_govt_consumption_in_lcu.sum() / total_monthly_value_added_in_lcu,
                )
            )

    @abstractmethod
    def set_gov_entity_total_consumption(
        self,
        monthly_govt_consumption_in_usd: np.ndarray,
        monthly_govt_consumption_in_lcu: np.ndarray,
        total_gov_consumption_growth: Optional[np.ndarray],
    ) -> None:
        pass
