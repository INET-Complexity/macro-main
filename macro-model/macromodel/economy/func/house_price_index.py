from abc import ABC
from typing import Optional

import numpy as np

from macromodel.forecaster.forecaster import (
    ConstantForecaster,
    ImplementedAutoregForecaster,
    ManualAutoregForecaster,
    OLSForecaster,
)


class HPIForecasting(ABC):
    def __init__(self, *args, **kwargs):
        self.forecaster = None

    def forecast_hpi_growth(
        self,
        historic_hpi: np.ndarray,
        min_hpi_growth: Optional[float] = None,
        max_hpi_growth: Optional[float] = None,
        t: int = 1,
        assume_zero_noise: bool = False,
    ) -> float:
        forecast = self.forecaster.forecast(
            historic_hpi,
            t,
            assume_zero_noise=assume_zero_noise,
        )
        if min_hpi_growth is not None:
            forecast = np.maximum(min_hpi_growth, forecast)
        if max_hpi_growth is not None:
            forecast = np.maximum(max_hpi_growth, forecast)
        return forecast


class HPIForecastingConstant(HPIForecasting):
    def __init__(self, value: float, *args, **kwargs):
        super().__init__()
        self.forecaster = ConstantForecaster(value=value)


class HPIForecastingOLS(HPIForecasting):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.forecaster = OLSForecaster()


class HPIImplementedForecastingAutoReg(HPIForecasting):
    def __init__(self, lags: int, *args, **kwargs):
        super().__init__()
        self.forecaster = ImplementedAutoregForecaster(lags)


class HPIManualForecastingAutoReg(HPIForecasting):
    def __init__(self, lags: int, *args, **kwargs):
        assert lags == 1
        super().__init__()
        self.forecaster = ManualAutoregForecaster()
