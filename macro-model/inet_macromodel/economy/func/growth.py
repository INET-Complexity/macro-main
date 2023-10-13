import numpy as np
import logging

from abc import abstractmethod, ABC

from inet_macromodel.forecaster.forecaster import (
    AutoregForecaster,
    OLSForecaster,
    ConstantForecaster,
)


class GrowthForecasting(ABC):
    @abstractmethod
    def forecast_growth(self, historic_growth: np.ndarray) -> float:
        pass


class GrowthForecastingConstant(GrowthForecasting):
    def __init__(self, value: float, *args, **kwargs):
        self.forecaster = ConstantForecaster(value=value)
        # if args and kwargs not empty, log warning
        if args or kwargs:
            logging.warning(
                "GrowthForecastingConstant: args and kwargs are not used. "
                "Please check the documentation."
            )

    def forecast_growth(self, historic_growth: np.ndarray) -> float:
        return self.forecaster.forecast(historic_growth)


class GrowthForecastingAutoReg(GrowthForecasting):
    def __init__(
        self, lags: int, window: int, use_log_output: bool = True, *args, **kwargs
    ):
        self.forecaster = AutoregForecaster(lags)
        self.window = window
        self.use_log_output = use_log_output
        # if args and kwargs not empty, log warning
        if args or kwargs:
            logging.warning(
                "GrowthForecastingAutoReg: args and kwargs are not used. "
                "Please check the documentation."
            )

    def forecast_growth(self, historic_growth: np.ndarray) -> float:
        historic_growth = historic_growth[-self.window :]
        if self.use_log_output:
            historic_output = np.cumprod(1 + historic_growth)
            forecast_output = np.exp(self.forecaster.forecast(np.log(historic_output)))
            return forecast_output / historic_output[-1] - 1.0
        else:
            return self.forecaster.forecast(historic_growth)


class GrowthForecastingOLS(GrowthForecasting):
    def __init__(self, window: int, *args, **kwargs):
        self.forecaster = OLSForecaster()
        self.window = window
        # if args and kwargs not empty, log warning
        if args or kwargs:
            logging.warning(
                "GrowthForecastingOLS: args and kwargs are not used. "
                "Please check the documentation."
            )

    def forecast_growth(self, historic_growth: np.ndarray) -> float:
        return self.forecaster.forecast(historic_growth[-self.window :])
