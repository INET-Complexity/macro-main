import numpy as np
from abc import ABC

from macromodel.forecaster.forecaster import (
    OLSForecaster,
    ConstantForecaster,
    ImplementedAutoregForecaster,
    ManualAutoregForecaster,
)


class GrowthForecasting(ABC):
    def __init__(self):
        self.forecaster = None

    def forecast_growth(
        self,
        historic_growth: np.ndarray,
        exogenous_growth: np.ndarray,
        current_time: int,
        t: int = 1,
        assume_zero_noise: bool = False,
    ) -> float:
        return self.forecaster.forecast(historic_growth, t, assume_zero_noise=assume_zero_noise)


class GrowthForecastingConstant(GrowthForecasting):
    def __init__(self, value: float):
        super().__init__()
        self.forecaster = ConstantForecaster(value=value)


class GrowthForecastingOLS(GrowthForecasting):
    def __init__(self):
        super().__init__()
        self.forecaster = OLSForecaster()


class GrowthImplementedForecastingAutoReg(GrowthForecasting):
    def __init__(self, lags: int):
        super().__init__()
        self.forecaster = ImplementedAutoregForecaster(lags)


class GrowthManualForecastingAutoReg(GrowthForecasting):
    def __init__(self, lags: int):
        assert lags == 1
        super().__init__()
        self.forecaster = ManualAutoregForecaster()


class ExogenousGrowthForecasting(GrowthForecasting):
    def __init__(self, lags: int):
        super().__init__()
        self.forecaster = None

    def forecast_growth(
        self,
        historic_growth: np.ndarray,
        exogenous_growth: np.ndarray,
        current_time: int,
        t: int = 1,
        assume_zero_noise: bool = False,
    ) -> float | np.ndarray:
        assert 0  # TODO: remove the outer exp(x) - 1 transform when using this
        return np.array([exogenous_growth[current_time]])
