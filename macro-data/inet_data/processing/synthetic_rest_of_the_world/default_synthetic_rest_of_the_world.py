import numpy as np

from sklearn.linear_model import LinearRegression

from inet_data.processing.synthetic_rest_of_the_world.synthetic_rest_of_the_world import (
    SyntheticRestOfTheWorld,
)

from typing import Optional


class DefaultSyntheticRestOfTheWorld(SyntheticRestOfTheWorld):
    def __init__(
        self,
        year: int,
    ):
        super().__init__(year=year)

    def create(
        self,
        row_imports: np.ndarray,
        row_exports: np.ndarray,
        exchange_rate_usd_to_lcu: float,
        row_exports_data_growth: Optional[np.ndarray],
        row_imports_data_growth: Optional[np.ndarray],
    ) -> None:
        self.set_imports(
            row_imports=row_imports,
            exchange_rate_usd_to_lcu=exchange_rate_usd_to_lcu,
            row_imports_data_growth=row_imports_data_growth,
        )
        self.set_exports(
            row_exports=row_exports,
            row_exports_data_growth=row_exports_data_growth,
        )
        self.set_prices(
            n_industries=len(row_imports),
            exchange_rate_usd_to_lcu=exchange_rate_usd_to_lcu,
        )

    def set_imports(
        self,
        row_imports: np.ndarray,
        exchange_rate_usd_to_lcu: float,
        row_imports_data_growth: Optional[np.ndarray],
    ) -> None:
        self.row_data["Imports in USD"] = row_imports
        self.row_data["Imports in LCU"] = exchange_rate_usd_to_lcu * row_imports
        if row_imports_data_growth is None:
            self.imports_model = None
        else:
            self.imports_model = LinearRegression().fit(
                [[0], [1]], [np.nanmean(row_imports_data_growth), np.nanmean(row_imports_data_growth)]
            )

    def set_exports(
        self,
        row_exports: np.ndarray,
        row_exports_data_growth: Optional[np.ndarray],
    ) -> None:
        self.row_data["Exports"] = row_exports
        if row_exports_data_growth is None:
            self.exports_model = None
        else:
            self.exports_model = LinearRegression().fit(
                [[0], [1]], [np.nanmean(row_exports_data_growth), np.nanmean(row_exports_data_growth)]
            )

    def set_prices(
        self,
        n_industries: int,
        exchange_rate_usd_to_lcu: float,
    ) -> None:
        self.row_data["Price in USD"] = np.ones(n_industries)
        self.row_data["Price in LCU"] = exchange_rate_usd_to_lcu * np.ones(n_industries)
