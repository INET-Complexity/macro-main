from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from inet_data.processing.synthetic_rest_of_the_world.synthetic_rest_of_the_world import (
    SyntheticRestOfTheWorld,
)
from inet_data.readers.default_readers import DataReaders


class DefaultSyntheticRestOfTheWorld(SyntheticRestOfTheWorld):
    def __init__(
        self,
        year: int,
        row_data: pd.DataFrame,
        exports_model: Optional[LinearRegression],
        imports_model: Optional[LinearRegression],
    ):
        super().__init__(year, row_data, exports_model, imports_model)

    @classmethod
    def init_from_readers(
        cls,
        year: int,
        readers: DataReaders,
        exogenous_row_data: Optional[dict[str, pd.DataFrame]],
        industry_data: dict[str, pd.DataFrame],
    ):
        if exogenous_row_data:
            row_exports_data = exogenous_row_data["iot_industry_data"].xs("Exports in USD", axis=1, level=0).sum(axis=1)
            row_exports_data = row_exports_data.loc[row_exports_data.index < pd.Timestamp(year, 1, 1)]
            row_exports_data_growth = (row_exports_data / row_exports_data.shift(1)).values
            if row_exports_data.isna().sum() >= 1:
                exports_model = LinearRegression().fit(
                    [[0], [1]], [np.nanmean(row_exports_data_growth), np.nanmean(row_exports_data_growth)]
                )
            else:
                exports_model = None
            row_imports_data = exogenous_row_data["iot_industry_data"].xs("Imports in USD", axis=1, level=0).sum(axis=1)
            row_imports_data = row_imports_data.loc[row_imports_data.index < pd.Timestamp(year, 1, 1)]
            row_imports_data_growth = (row_imports_data / row_imports_data.shift(1)).values
            if row_imports_data.isna().sum() >= 1:
                imports_model = LinearRegression().fit(
                    [[0], [1]], [np.nanmean(row_imports_data_growth), np.nanmean(row_imports_data_growth)]
                )
            else:
                imports_model = None
        else:
            exports_model = None
            imports_model = None

        row_exports = industry_data["ROW"]["industry_vectors"]["Exports in USD"]
        row_imports = industry_data["ROW"]["industry_vectors"]["Imports in USD"]
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

        return cls(year, row_data, exports_model, imports_model)
