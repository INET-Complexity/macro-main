from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from macro_data.configuration.countries import Country
from macro_data.processing.synthetic_government_entities.synthetic_government_entities import (
    SyntheticGovernmentEntities,
)
from macro_data.readers.default_readers import DataReaders


class DefaultSyntheticGovernmentEntities(SyntheticGovernmentEntities):
    """
    Represents a collection of synthetic government entities. These entities are used to represent government consumption.

    Parameters:
    - country_name (str): The name of the country.
    - year (int): The year of the data.
    - number_of_entities (int): The number of government entities.
    - gov_entity_data (pd.DataFrame): The data for the government entities.
    - government_consumption_model (Optional[LinearRegression]): The consumption model for the government (a linear
    regression model to extrapolate government consumption growth).

    Attributes:
    - country_name (str): The name of the country.
    - year (int): The year of the data.
    - number_of_entities (int): The number of government entities.
    - gov_entity_data (pd.DataFrame): The data for the government entities.
    - government_consumption_model (Optional[LinearRegression]): The consumption model for the government.


    Methods:
    - create_from_readers: Creates an instance of SyntheticDefaultGovernmentEntities from data readers.

    """

    def __init__(
        self,
        country_name: Country,
        year: int,
        number_of_entities: int,
        gov_entity_data: pd.DataFrame,
        government_consumption_model: Optional[LinearRegression] = None,
    ):
        super().__init__(
            country_name,
            year,
            number_of_entities,
            gov_entity_data,
            government_consumption_model,
        )

    @classmethod
    def from_readers(
        cls,
        readers: DataReaders,
        country_name: Country,
        year: int,
        exogenous_country_data: Optional[dict[str, pd.DataFrame]],
        industry_data: dict[str, pd.DataFrame],
        single_government_entity: bool,
    ):
        create_model = False
        if exogenous_country_data:
            total_gov_consumption = (
                exogenous_country_data["iot_industry_data"]
                .xs("Government Consumption in USD", axis=1, level=0)
                .sum(axis=1)
            )
            total_gov_consumption_filtered = total_gov_consumption.loc[:f"{year}-01-01":]
            growth_series = total_gov_consumption_filtered / total_gov_consumption_filtered.shift(1)
            total_gov_consumption_growth = growth_series.values
            if growth_series.dropna().shape[0] > 0:
                create_model = True
        else:
            total_gov_consumption_growth = None

        monthly_govt_consumption_in_usd = industry_data["industry_vectors"]["Government Consumption in USD"].values
        monthly_govt_consumption_in_lcu = industry_data["industry_vectors"]["Government Consumption in LCU"].values
        total_monthly_va_lcu = industry_data["industry_vectors"]["Value Added in LCU"].sum()
        total_number_of_firms = int(
            readers.oecd_econ.read_business_demography(
                country=country_name,
                output=pd.Series(industry_data["industry_vectors"]["Output in LCU"].values),
                year=year,
            ).sum()
        )

        n_entities = int(
            max(
                1,
                total_number_of_firms * monthly_govt_consumption_in_lcu.sum() / total_monthly_va_lcu,
            )
        )
        n_entities = 1 if single_government_entity else n_entities

        gov_entity_data = pd.DataFrame(
            {
                "Consumption in USD": monthly_govt_consumption_in_usd,
                "Consumption in LCU": monthly_govt_consumption_in_lcu,
            }
        )

        if create_model:
            government_consumption_model = LinearRegression().fit(
                [[0], [1]], [np.nanmean(total_gov_consumption_growth), np.nanmean(total_gov_consumption_growth)]
            )
        else:
            government_consumption_model = None

        return cls(
            country_name,
            year,
            n_entities,
            gov_entity_data,
            government_consumption_model,
        )
