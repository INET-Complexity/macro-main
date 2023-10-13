import numpy as np
import logging

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
    def __init__(self, value: float, *args, **kwargs):
        self.forecaster = ConstantForecaster(value=value)
        if args or kwargs:
            logging.warning(
                "InflationForecastingConstant: args and kwargs are not used. "
                "Please check the documentation."
            )

    def forecast_inflation(self, historic_inflation: np.ndarray) -> float:
        return self.forecaster.forecast(historic_inflation)


class InflationForecastingAutoReg(InflationForecasting):
    def __init__(self, lags: int, window: int, *args, **kwargs):
        self.forecaster = AutoregForecaster(lags)
        self.window = window
        if args or kwargs:
            logging.warning(
                "InflationForecastingAutoReg: args and kwargs are not used. "
                "Please check the documentation."
            )

    def forecast_inflation(self, historic_inflation: np.ndarray) -> float:
        return self.forecaster.forecast(historic_inflation[-self.window :])


class InflationForecastingOLS(InflationForecasting):
    def __init__(self, window: int, *args, **kwargs):
        self.forecaster = OLSForecaster()
        self.window = window
        if args or kwargs:
            logging.warning(
                "InflationForecastingOLS: args and kwargs are not used. "
                "Please check the documentation."
            )

    def forecast_inflation(self, historic_inflation: np.ndarray) -> float:
        return self.forecaster.forecast(historic_inflation[-self.window :])
