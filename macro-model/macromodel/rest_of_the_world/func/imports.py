from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from macromodel.forecaster.forecaster import \
    ImplementedAutoregForecaster  # noqa
from macromodel.forecaster.forecaster import ManualAutoregForecaster


class RoWImportsSetter(ABC):
    def __init__(self, consistency: float):
        self.consistency = max(0.0, min(1.0, consistency))
        self.fixed_total_imports = None

        assert self.consistency == 0.0 or self.consistency == 1.0

    @abstractmethod
    def compute_imports(
        self,
        historic_total_real_imports: np.ndarray,
        historic_total_real_imports_during: np.ndarray,
        current_time: int,
        initial_desired_imports: np.ndarray,
        model: Optional[Any],
        aggregate_country_production_index: float,
        aggregate_country_price_index: float,
        adjustment_speed: float,
        assume_zero_noise: bool,
    ) -> np.ndarray:
        pass


class AutoregressiveRoWImportsSetter(RoWImportsSetter):
    def compute_imports(
        self,
        historic_total_real_imports: np.ndarray,
        historic_total_real_imports_during: np.ndarray,
        current_time: int,
        initial_desired_imports: np.ndarray,
        model: Optional[Any],
        aggregate_country_production_index: float,
        aggregate_country_price_index: float,
        adjustment_speed: float,
        assume_zero_noise: bool,
    ) -> np.ndarray:
        if historic_total_real_imports[-1] == 0.0:
            return np.zeros(initial_desired_imports.shape)

        # Fitting based on target exports
        if self.consistency == 1.0:
            if self.fixed_total_imports is None:
                self.fixed_total_imports = np.exp(
                    ManualAutoregForecaster().forecast(
                        data=np.log(historic_total_real_imports[: -current_time - 1]),
                        t=20,
                        assume_zero_noise=assume_zero_noise,
                    )
                )
            imports = self.fixed_total_imports[current_time]  # check!

        # Fitting based on historic consumption
        else:
            imports = np.exp(
                ManualAutoregForecaster().forecast(
                    data=np.log(historic_total_real_imports),
                    t=1,
                    assume_zero_noise=assume_zero_noise,
                )[0]
            )

        # Weighted by prices
        return np.maximum(
            0.0,
            (1.0 + adjustment_speed * (aggregate_country_price_index - 1.0))
            * imports
            * initial_desired_imports
            / initial_desired_imports.sum(),
        )


class InflationRoWImportsSetter(RoWImportsSetter):
    def compute_imports(
        self,
        historic_total_real_imports: np.ndarray,
        historic_total_real_imports_during: np.ndarray,
        current_time: int,
        initial_desired_imports: np.ndarray,
        model: Optional[Any],
        aggregate_country_production_index: float,
        aggregate_country_price_index: float,
        adjustment_speed: float,
        assume_zero_noise: bool,
    ) -> np.ndarray:
        return np.maximum(
            0.0,
            (1.0 + adjustment_speed * (aggregate_country_production_index - 1.0))
            * (1.0 + adjustment_speed * (aggregate_country_price_index - 1.0))
            * initial_desired_imports,
        )


class ExogenousRoWImportsSetter(RoWImportsSetter):
    def compute_imports(
        self,
        historic_total_real_imports: np.ndarray,
        historic_total_real_imports_during: np.ndarray,
        current_time: int,
        initial_desired_imports: np.ndarray,
        model: Optional[Any],
        aggregate_country_production_index: float,
        aggregate_country_price_index: float,
        adjustment_speed: float,
        assume_zero_noise: bool,
    ) -> np.ndarray:
        return (
            aggregate_country_price_index
            * historic_total_real_imports_during[current_time]
            * initial_desired_imports
            / initial_desired_imports.sum()
        )


"""
class DefaultRoWImportsSetter(RoWImportsSetter):
    def compute_imports(
        self,
        initial_desired_imports: np.ndarray,
        model: Optional[Any],
        aggregate_country_production_index: float,
        aggregate_country_price_index: float,
        adjustment_speed: float,
    ) -> np.ndarray:
        if model is None:
            return initial_desired_imports
        return np.maximum(
            0.0, model.predict([[0]])[0] * initial_desired_imports
        )
"""
