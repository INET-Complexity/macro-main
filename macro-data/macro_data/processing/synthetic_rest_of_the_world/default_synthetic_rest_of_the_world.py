from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from macro_data.configuration.dataconfiguration import ROWDataConfiguration
from macro_data.processing.synthetic_rest_of_the_world.synthetic_rest_of_the_world import (
    SyntheticRestOfTheWorld,
)
from macro_data.readers.default_readers import DataReaders


class DefaultSyntheticRestOfTheWorld(SyntheticRestOfTheWorld):
    def __init__(
        self,
        year: int,
        row_data: pd.DataFrame,
        n_exporters_by_industry: np.ndarray,
        n_importers,
        exports_model: Optional[LinearRegression],
        imports_model: Optional[LinearRegression],
    ):
        super().__init__(
            year=year,
            row_data=row_data,
            n_exporters_by_industry=n_exporters_by_industry,
            exports_model=exports_model,
            imports_model=imports_model,
            n_importers=n_importers,
        )

    @classmethod
    def from_readers(
        cls,
        year: int,
        readers: DataReaders,
        industry_data: dict[str, dict[str, pd.DataFrame]],
        n_sellers_by_industry: np.ndarray,
        n_buyers: int,
        row_configuration: ROWDataConfiguration,
        row_exports_growth: Optional[pd.Series] = None,
        row_imports_growth: Optional[pd.Series] = None,
    ):
        row_industry_data = industry_data["ROW"]

        total_imports = sum(
            [industry_data[c]["industry_vectors"]["Imports in USD"].sum() for c in industry_data if c != "ROW"]
        )
        exports_by_industry = np.sum(
            [industry_data[c]["industry_vectors"]["Exports in USD"].values for c in industry_data if c != "ROW"], axis=1
        )

        row_exports = row_industry_data["industry_vectors"]["Exports in USD"]
        row_imports = row_industry_data["industry_vectors"]["Imports in USD"]
        exchange_rate = readers.exchange_rates.from_usd_to_lcu("ROW", year)

        row_data = pd.DataFrame(
            {
                "Exports": row_exports,
                "Imports in USD": row_imports,
                "Imports in LCU": exchange_rate * row_imports,
            }
        )

        row_data["Price in USD"] = 1
        row_data["Price in LCU"] = exchange_rate * row_data["Price in USD"]

        if row_configuration.model_exports:
            if row_exports_growth is None:
                raise ValueError("Exports growth data is required.")
            exports_model = LinearRegression()
            exports_model.fit([[0], [1]], [row_exports_growth.mean(), row_exports_growth.mean()])
        else:
            exports_model = None

        if row_configuration.model_imports:
            if row_imports_growth is None:
                raise ValueError("Imports growth data is required.")
            imports_model = LinearRegression()
            imports_model.fit([[0], [1]], [row_imports_growth.mean(), row_imports_growth.mean()])
        else:
            imports_model = None

        if row_configuration.assume_one_exporter_by_industry:
            n_exporters_by_industry = np.ones(row_data.shape[0])
        else:
            n_exporters_by_industry = np.maximum(
                1, row_data["Exports"] / exports_by_industry * n_sellers_by_industry
            ).astype(int)

        n_importers = int(max(1, row_data["Imports in USD"].sum() / total_imports * n_buyers))

        return cls(
            year=year,
            row_data=row_data,
            n_exporters_by_industry=n_exporters_by_industry,
            exports_model=exports_model,
            imports_model=imports_model,
            n_importers=n_importers,
        )
