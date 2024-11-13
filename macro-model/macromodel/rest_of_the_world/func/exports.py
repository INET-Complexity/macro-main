from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from macromodel.forecaster.forecaster import ImplementedAutoregForecaster  # noqa
from macromodel.forecaster.forecaster import ManualAutoregForecaster


class RoWExportsSetter(ABC):
    def __init__(self, consistency: float):
        self.consistency = consistency
        self.fixed_total_exports = None

        assert self.consistency == 0.0 or self.consistency == 1.0

    @abstractmethod
    def compute_exports(
        self,
        historic_total_real_exports: np.ndarray,
        historic_total_real_exports_during: np.ndarray,
        current_time: int,
        initial_desired_exports: np.ndarray,
        model: Optional[Any],
        aggregate_country_production_index: float,
        adjustment_speed: float,
        assume_zero_noise: bool,
    ) -> np.ndarray:
        pass


class AutoregressiveRoWExportsSetter(RoWExportsSetter):
    def compute_exports(
        self,
        historic_total_real_exports: np.ndarray,
        historic_total_real_exports_during: np.ndarray,
        current_time: int,
        initial_desired_exports: np.ndarray,
        model: Optional[Any],
        aggregate_country_production_index: float,
        adjustment_speed: float,
        assume_zero_noise: bool,
    ) -> np.ndarray:
        if historic_total_real_exports[-1] == 0.0:
            return np.zeros(initial_desired_exports.shape)

        # Fitting based on target exports
        if self.consistency == 1.0:
            if self.fixed_total_exports is None:
                self.fixed_total_exports = np.exp(
                    ManualAutoregForecaster().forecast(
                        data=np.log(historic_total_real_exports[: -current_time - 1]),
                        t=20,
                        assume_zero_noise=assume_zero_noise,
                    )
                )
            exports = self.fixed_total_exports[current_time]  # check!

        # Fitting based on historic consumption
        else:
            exports = np.exp(
                ManualAutoregForecaster().forecast(
                    data=np.log(historic_total_real_exports),
                    t=1,
                    assume_zero_noise=assume_zero_noise,
                )[0]
            )

        # Weighted by prices
        return np.maximum(
            0.0,
            exports * initial_desired_exports / initial_desired_exports.sum(),
        )


class GrowthRoWExportsSetter(RoWExportsSetter):
    def compute_exports(
        self,
        historic_total_real_exports: np.ndarray,
        historic_total_real_exports_during: np.ndarray,
        current_time: int,
        initial_desired_exports: np.ndarray,
        model: Optional[Any],
        aggregate_country_production_index: float,
        adjustment_speed: float,
        assume_zero_noise: bool,
    ) -> np.ndarray:
        return np.maximum(
            0.0,
            (1.0 + adjustment_speed * (aggregate_country_production_index - 1.0)) * initial_desired_exports,
        )


class ExogenousRoWExportsSetter(RoWExportsSetter):
    def compute_exports(
        self,
        historic_total_real_exports: np.ndarray,
        historic_total_real_exports_during: np.ndarray,
        current_time: int,
        initial_desired_exports: np.ndarray,
        model: Optional[Any],
        aggregate_country_production_index: float,
        adjustment_speed: float,
        assume_zero_noise: bool,
    ) -> np.ndarray:
        return (
            historic_total_real_exports_during[current_time] * initial_desired_exports / initial_desired_exports.sum()
        )


"""
class DefaultRoWExportsSetter(RoWExportsSetter):
    def compute_exports(
        self,
        historic_total_real_exports: np.ndarray,
        historic_total_real_exports_during: np.ndarray,
        initial_desired_exports: np.ndarray,
        model: Optional[Any],
        aggregate_country_production_index: float,
        adjustment_speed: float,
    ) -> np.ndarray:
        if model is None:
            return initial_desired_exports
        return np.maximum(
            0.0, model.predict([[0]])[0] * initial_desired_exports
        )
"""
