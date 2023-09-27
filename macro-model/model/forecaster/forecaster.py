import numpy as np
import statsmodels.api as sm

from abc import ABC, abstractmethod
from statsmodels.tsa.ar_model import AutoReg

from typing import Literal


def check_len(data: np.ndarray):
    if len(data) < 3:
        raise ValueError("Array is too small to be forecasted")


class Forecaster(ABC):
    @abstractmethod
    def forecast(self, ts: np.ndarray) -> float:
        pass


class AutoregForecaster(Forecaster):
    def __init__(self, lags: int, trend: Literal["n", "c", "t", "ct"] = "ct"):
        self.lags = lags
        self.trend = trend

    def forecast(self, data: np.ndarray, t: int = 1) -> float:
        check_len(data)
        forecast = AutoReg(data, lags=[self.lags], trend=self.trend, old_names=False).fit().forecast(steps=t)
        return float(forecast)


class OLSForecaster(Forecaster):
    def forecast(self, data: np.ndarray, t: int = 1) -> float:
        check_len(data)
        x = sm.add_constant(np.arange(len(data)))
        model = sm.OLS(data, x).fit()
        prediction = model.params[0] + model.params[1] * len(data)
        return prediction


class ConstantForecaster(Forecaster):
    def __init__(self, value: float):
        self.value = value

    def forecast(self, data: np.ndarray) -> float:
        return self.value
