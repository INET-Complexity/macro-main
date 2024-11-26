from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from macromodel.forecaster.forecaster import ImplementedAutoregForecaster  # noqa
from macromodel.forecaster.forecaster import ManualAutoregForecaster


class GovernmentConsumptionSetter(ABC):
    def __init__(
        self,
        consistency: float,
        default_growth: Optional[float] = None,
    ):
        assert consistency == 1.0 or consistency == 0.0
        self.consistency = consistency
        self.default_growth = default_growth
        self.fixed_total_government_consumption = None
        self.buffer = 20

    @abstractmethod
    def compute_target_consumption(
        self,
        previous_desired_government_consumption: np.ndarray,
        model: Optional[Any],
        historic_total_consumption: np.ndarray,
        initial_good_prices: np.ndarray,
        current_good_prices: np.ndarray,
        expected_growth: float,
        expected_inflation: float,
        current_time: int,
        exogenous_total_consumption: Optional[np.ndarray],
        forecasting_window: int,
        assume_zero_noise: bool = False,
    ) -> np.ndarray:
        pass


class AutoregressiveGovernmentConsumptionSetter(GovernmentConsumptionSetter):
    def compute_target_consumption(
        self,
        previous_desired_government_consumption: np.ndarray,
        model: Optional[Any],
        historic_total_consumption: np.ndarray,
        initial_good_prices: np.ndarray,
        current_good_prices: np.ndarray,
        expected_growth: float,
        expected_inflation: float,
        current_time: int,
        exogenous_total_consumption: Optional[np.ndarray],
        forecasting_window: int,
        assume_zero_noise: bool = False,
        log_it: bool = True,
    ) -> np.ndarray:
        if historic_total_consumption[-1] == 0.0:
            return np.zeros(previous_desired_government_consumption.shape)

        # Fitting based on target consumption
        if self.consistency == 1.0:
            if (
                self.fixed_total_government_consumption is None
                or len(self.fixed_total_government_consumption) < current_time
            ):
                if log_it:
                    self.fixed_total_government_consumption = np.exp(
                        ManualAutoregForecaster().forecast(
                            data=np.log(historic_total_consumption),
                            t=max(current_time + self.buffer, current_time),
                            assume_zero_noise=assume_zero_noise,
                        )
                    )
                else:
                    self.fixed_total_government_consumption = ManualAutoregForecaster().forecast(
                        data=historic_total_consumption,
                        t=max(current_time + self.buffer, current_time),
                        assume_zero_noise=assume_zero_noise,
                    )
            consumption = self.fixed_total_government_consumption[current_time - 1]

        # Fitting based on historic consumption
        else:
            consumption = np.exp(
                ManualAutoregForecaster().forecast(
                    data=np.log(historic_total_consumption),
                    t=1,
                    assume_zero_noise=assume_zero_noise,
                )[0]
            )

        # Weighted by prices
        return np.maximum(
            0.0,
            (1 + expected_inflation)
            * current_good_prices
            / initial_good_prices
            * consumption
            * previous_desired_government_consumption
            / previous_desired_government_consumption.sum(),
        )


class ConstantGrowthGovernmentConsumptionSetter(GovernmentConsumptionSetter):
    def compute_target_consumption(
        self,
        previous_desired_government_consumption: np.ndarray,
        model: Optional[Any],
        historic_total_consumption: Optional[np.ndarray],
        initial_good_prices: np.ndarray,
        current_good_prices: np.ndarray,
        expected_growth: float,
        expected_inflation: float,
        current_time: int,
        exogenous_total_consumption: Optional[np.ndarray],
        forecasting_window: int,
        assume_zero_noise: bool = False,
    ) -> np.ndarray:
        if historic_total_consumption is None:
            return np.maximum(
                0.0,
                (1 + expected_inflation) * (1 + self.default_growth) * previous_desired_government_consumption,
            )
        if self.default_growth is None:
            self.default_growth = np.mean(
                np.log(
                    historic_total_consumption[1 : -current_time - 1]
                    / historic_total_consumption[0 : -current_time - 2]
                )
            )

        return np.maximum(
            0.0,
            (1 + expected_inflation)
            * current_good_prices
            / initial_good_prices
            * (1 + self.default_growth)
            * previous_desired_government_consumption,
        )


class AutoregressiveGrowthGovernmentConsumptionSetter(GovernmentConsumptionSetter):
    def compute_target_consumption(
        self,
        previous_desired_government_consumption: np.ndarray,
        model: Optional[Any],
        historic_total_consumption: np.ndarray,
        initial_good_prices: np.ndarray,
        current_good_prices: np.ndarray,
        expected_growth: float,
        expected_inflation: float,
        current_time: int,
        exogenous_total_consumption: Optional[np.ndarray],
        forecasting_window: int,
        assume_zero_noise: bool = False,
        log_it: bool = False,
    ) -> np.ndarray:
        if historic_total_consumption[-1] == 0.0:
            return np.zeros(previous_desired_government_consumption.shape)

        # Fitting based on target consumption
        if self.consistency == 1.0:
            if self.fixed_total_government_consumption is None:
                historic_total_consumption_growth = (
                    historic_total_consumption[1:] / historic_total_consumption[:-1] - 1.0
                )
                self.fixed_total_government_consumption = (
                    np.exp(
                        ManualAutoregForecaster().forecast(
                            data=historic_total_consumption_growth,
                            t=20,
                            assume_zero_noise=assume_zero_noise,
                        )
                    )
                    - 1
                )
                self.fixed_total_government_consumption = (
                    np.cumprod(1 + self.fixed_total_government_consumption) * historic_total_consumption[-1]
                )

            consumption = self.fixed_total_government_consumption[current_time - 1]

        # Fitting based on historic consumption
        else:
            consumption = np.exp(
                ManualAutoregForecaster().forecast(
                    data=np.log(historic_total_consumption),
                    t=1,
                    assume_zero_noise=assume_zero_noise,
                )[0]
            )

        # Weighted by prices
        return np.maximum(
            0.0,
            (1 + expected_inflation)
            * current_good_prices
            / initial_good_prices
            * consumption
            * previous_desired_government_consumption
            / previous_desired_government_consumption.sum(),
        )


class ExogenousGovernmentConsumptionSetter(GovernmentConsumptionSetter):
    def compute_target_consumption(
        self,
        previous_desired_government_consumption: np.ndarray,
        model: Optional[Any],
        historic_total_consumption: Optional[np.ndarray],
        initial_good_prices: np.ndarray,
        current_good_prices: np.ndarray,
        expected_growth: float,
        expected_inflation: float,
        current_time: int,
        exogenous_total_consumption: Optional[np.ndarray],
        forecasting_window: int,
        assume_zero_noise: bool = False,
    ) -> np.ndarray:
        # print("GOV", exogenous_total_consumption)
        if exogenous_total_consumption is None:
            return np.maximum(
                0.0,
                (1 + expected_inflation)
                * current_good_prices
                / initial_good_prices
                * (1 + self.default_growth)
                * previous_desired_government_consumption,
            )
        if current_time >= len(exogenous_total_consumption):
            raise ValueError("No exogenous data available beyond this point.")
        return (
            (1 + expected_inflation)
            * current_good_prices
            / initial_good_prices
            * exogenous_total_consumption[current_time]
            * previous_desired_government_consumption
            / previous_desired_government_consumption.sum()
        )
