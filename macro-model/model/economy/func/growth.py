import numpy as np

from abc import abstractmethod, ABC

from model.forecaster.forecaster import (
    AutoregForecaster,
    OLSForecaster,
    ConstantForecaster,
)


class GrowthForecasting(ABC):
    @abstractmethod
    def forecast_growth(self, historic_growth: np.ndarray) -> float:
        pass


class GrowthForecastingConstant(GrowthForecasting):
    def __init__(self, value: float):
        self.forecaster = ConstantForecaster(value=value)

    def forecast_growth(self, historic_growth: np.ndarray) -> float:
        return self.forecaster.forecast(historic_growth)


class GrowthForecastingAutoReg(GrowthForecasting):
    def __init__(self, lags: int, window: int, use_log_output: bool = True):
        self.forecaster = AutoregForecaster(lags)
        self.window = window
        self.use_log_output = use_log_output

    def forecast_growth(self, historic_growth: np.ndarray) -> float:
        historic_growth = historic_growth[-self.window :]
        if self.use_log_output:
            historic_output = np.cumprod(1 + historic_growth)
            forecast_output = np.exp(self.forecaster.forecast(np.log(historic_output)))
            return forecast_output / historic_output[-1] - 1.0
        else:
            return self.forecaster.forecast(historic_growth)


class GrowthForecastingOLS(GrowthForecasting):
    def __init__(self, window: int):
        self.forecaster = OLSForecaster()
        self.window = window

    def forecast_growth(self, historic_growth: np.ndarray) -> float:
        return self.forecaster.forecast(historic_growth[-self.window :])
