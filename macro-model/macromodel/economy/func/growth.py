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
        t: int = 1,
        assume_zero_noise: bool = False,
    ) -> float:
        return self.forecaster.forecast(historic_growth, t, assume_zero_noise=assume_zero_noise)


class GrowthForecastingConstant(GrowthForecasting):
    def __init__(self, value: float, *args, **kwargs):
        super().__init__()
        self.forecaster = ConstantForecaster(value=value)


class GrowthForecastingOLS(GrowthForecasting):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.forecaster = OLSForecaster()


class GrowthImplementedForecastingAutoReg(GrowthForecasting):
    def __init__(self, lags: int, *args, **kwargs):
        super().__init__()
        self.forecaster = ImplementedAutoregForecaster(lags)


class GrowthManualForecastingAutoReg(GrowthForecasting):
    def __init__(self, lags: int, *args, **kwargs):
        assert lags == 1
        super().__init__()
        self.forecaster = ManualAutoregForecaster()
