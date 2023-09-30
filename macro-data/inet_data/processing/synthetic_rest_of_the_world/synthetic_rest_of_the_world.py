from abc import abstractmethod, ABC

import numpy as np
import pandas as pd


class SyntheticRestOfTheWorld(ABC):
    @abstractmethod
    def __init__(
        self,
        year: int,
    ):
        self.country_name = "ROW"
        self.year = year

        # Rest of the World inet_data
        self.row_data = pd.DataFrame()

        # Models
        self.exports_model = None
        self.imports_model = None

    @abstractmethod
    def create(
        self,
        row_imports: np.ndarray,
        row_exports: np.ndarray,
        exchange_rate_usd_to_lcu: float,
        row_exports_data_growth: np.ndarray,
        row_imports_data_growth: np.ndarray,
    ) -> None:
        pass

    @abstractmethod
    def set_imports(
        self,
        row_imports: np.ndarray,
        exchange_rate_usd_to_lcu: float,
        row_imports_data_growth: np.ndarray,
    ) -> None:
        pass

    @abstractmethod
    def set_exports(
        self,
        row_exports: np.ndarray,
        row_exports_data_growth: np.ndarray,
    ) -> None:
        pass

    @abstractmethod
    def set_prices(
        self,
        n_industries: int,
        exchange_rate_usd_to_lcu: float,
    ) -> None:
        pass
