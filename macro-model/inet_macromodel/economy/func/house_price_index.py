import logging
import numpy as np
from abc import abstractmethod, ABC

from inet_macromodel.forecaster.forecaster import (
    AutoregForecaster,
    OLSForecaster,
    ConstantForecaster,
)


class HPIForecasting(ABC):
    @abstractmethod
    def forecast_hpi_growth(self, historic_hpi_growth: np.ndarray) -> float:
        pass


class HPIForecastingConstant(HPIForecasting):
    def __init__(self, value: float, *args, **kwargs):
        self.forecaster = ConstantForecaster(value=value)
        if args or kwargs:
            logging.warning("HPIForecastingConstant: args and kwargs are not used. " "Please check the documentation.")

    def forecast_hpi_growth(self, historic_hpi_growth: np.ndarray) -> float:
        return self.forecaster.forecast(historic_hpi_growth)


class HPIForecastingAutoReg(HPIForecasting):
    def __init__(self, lags: int, window: int, *args, **kwargs):
        self.forecaster = AutoregForecaster(lags)
        self.window = window
        if args or kwargs:
            logging.warning("HPIForecastingAutoReg: args and kwargs are not used. " "Please check the documentation.")

    def forecast_hpi_growth(self, historic_hpi_growth: np.ndarray) -> float:
        return self.forecaster.forecast(historic_hpi_growth[-self.window :])


class HPIForecastingOLS(HPIForecasting):
    def __init__(self, window: int, *args, **kwargs):
        self.forecaster = OLSForecaster()
        self.window = window
        if args or kwargs:
            logging.warning("HPIForecastingOLS: args and kwargs are not used. " "Please check the documentation.")

    def forecast_hpi_growth(self, historic_hpi_growth: np.ndarray) -> float:
        return self.forecaster.forecast(historic_hpi_growth[-self.window :])
