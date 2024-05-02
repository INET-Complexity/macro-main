from typing import Optional

import numpy as np
from abc import ABC

from macromodel.forecaster.forecaster import (
    OLSForecaster,
    ConstantForecaster,
    ImplementedAutoregForecaster,
    ManualAutoregForecaster,
)


class InflationForecasting(ABC):
    def __init__(self):
        self.forecaster = None

    def forecast_inflation(
        self,
        historic_inflation: np.ndarray,
        min_inflation: Optional[float] = None,
        max_inflation: Optional[float] = None,
        t: int = 1,
        assume_zero_noise: bool = False,
    ) -> float | np.ndarray:
        forecast = self.forecaster.forecast(
            historic_inflation,
            t,
            assume_zero_noise=assume_zero_noise,
        )
        if min_inflation is not None:
            forecast = np.maximum(min_inflation, forecast)
        if max_inflation is not None:
            forecast = np.maximum(max_inflation, forecast)
        return forecast


class InflationForecastingConstant(InflationForecasting):
    def __init__(self, value: float, *args, **kwargs):
        super().__init__()
        self.forecaster = ConstantForecaster(value=value)


class InflationForecastingOLS(InflationForecasting):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.forecaster = OLSForecaster()


class InflationImplementedForecastingAutoReg(InflationForecasting):
    def __init__(self, lags: int, *args, **kwargs):
        super().__init__()
        self.forecaster = ImplementedAutoregForecaster(lags)


class InflationManualForecastingAutoReg(InflationForecasting):
    def __init__(self, lags: int, *args, **kwargs):
        assert lags == 1
        super().__init__()
        self.forecaster = ManualAutoregForecaster()
