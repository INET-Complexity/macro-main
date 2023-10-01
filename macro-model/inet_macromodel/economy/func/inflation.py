import numpy as np

from abc import abstractmethod, ABC

from inet_macromodel.forecaster.forecaster import (
    AutoregForecaster,
    OLSForecaster,
    ConstantForecaster,
)


class InflationForecasting(ABC):
    @abstractmethod
    def forecast_inflation(self, historic_inflation: np.ndarray) -> float:
        pass


class InflationForecastingConstant(InflationForecasting):
    def __init__(self, value: float):
        self.forecaster = ConstantForecaster(value=value)

    def forecast_inflation(self, historic_inflation: np.ndarray) -> float:
        return self.forecaster.forecast(historic_inflation)


class InflationForecastingAutoReg(InflationForecasting):
    def __init__(self, lags: int, window: int):
        self.forecaster = AutoregForecaster(lags)
        self.window = window

    def forecast_inflation(self, historic_inflation: np.ndarray) -> float:
        return self.forecaster.forecast(historic_inflation[-self.window :])


class InflationForecastingOLS(InflationForecasting):
    def __init__(self, window: int):
        self.forecaster = OLSForecaster()
        self.window = window

    def forecast_inflation(self, historic_inflation: np.ndarray) -> float:
        return self.forecaster.forecast(historic_inflation[-self.window :])
