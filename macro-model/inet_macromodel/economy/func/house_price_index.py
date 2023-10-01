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
    def __init__(self, value: float):
        self.forecaster = ConstantForecaster(value=value)

    def forecast_hpi_growth(self, historic_hpi_growth: np.ndarray) -> float:
        return self.forecaster.forecast(historic_hpi_growth)


class HPIForecastingAutoReg(HPIForecasting):
    def __init__(self, lags: int, window: int):
        self.forecaster = AutoregForecaster(lags)
        self.window = window

    def forecast_hpi_growth(self, historic_hpi_growth: np.ndarray) -> float:
        return self.forecaster.forecast(historic_hpi_growth[-self.window :])


class HPIForecastingOLS(HPIForecasting):
    def __init__(self, window: int):
        self.forecaster = OLSForecaster()
        self.window = window

    def forecast_hpi_growth(self, historic_hpi_growth: np.ndarray) -> float:
        return self.forecaster.forecast(historic_hpi_growth[-self.window :])
