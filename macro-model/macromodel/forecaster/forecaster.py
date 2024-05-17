import numpy as np
import statsmodels.api as sm

from abc import ABC, abstractmethod
from statsmodels.tsa.ar_model import AutoReg

from typing import Literal


def check_len(data: np.ndarray):
    if len(data) < 3:
        raise ValueError("Array is too small to be forecasted", data)


class Forecaster(ABC):
    @abstractmethod
    def forecast(self, data: np.ndarray, t: int = 1, assume_zero_noise: bool = False) -> float:
        pass


class ConstantForecaster(Forecaster):
    def __init__(self, value: float):
        self.value = value

    def forecast(self, data: np.ndarray, t: int = 1, assume_zero_noise: bool = False) -> np.ndarray:
        return np.array([self.value])


class OLSForecaster(Forecaster):
    def forecast(self, data: np.ndarray, t: int = 1, assume_zero_noise: bool = False) -> np.ndarray:
        check_len(data)
        x = sm.add_constant(np.arange(len(data)))
        model = sm.OLS(data, x).fit()
        prediction = model.params[0] + model.params[1] * len(data)
        return np.array([prediction])


class ImplementedAutoregForecaster(Forecaster):
    def __init__(self, lags: int, trend: Literal["n", "c", "t", "ct"] = "t"):
        self.lags = lags
        self.trend = trend

    def forecast(self, data: np.ndarray, t: int = 1, assume_zero_noise: bool = False) -> float | np.ndarray:
        check_len(data)
        forecast = AutoReg(data, lags=[self.lags], trend=self.trend, old_names=False).fit().forecast(steps=t)
        return forecast


class ManualAutoregForecaster(Forecaster):
    def forecast(
        self,
        data: np.ndarray,
        t: int = 1,
        assume_zero_noise: bool = False,
    ) -> np.ndarray:
        data = data[~np.isnan(data)]
        check_len(data)
        var = self.rfvar3(data, np.ones((len(data), 1)))
        if assume_zero_noise:
            eps = 0.0
        else:
            eps = np.random.normal(0.0, np.sqrt(np.var(var["u"])))
        vals = [var["By"][0][0][0] * data[-1] + var["Bx"][0] + eps]
        for t_ in range(t - 1):
            if assume_zero_noise:
                eps = 0.0
            else:
                eps = np.random.normal(0.0, np.sqrt(np.var(var["u"])))
            vals.append(var["By"][0][0][0] * vals[-1] + var["Bx"][0] + eps)
        return np.array(vals)

    def get_noise_variance(self, data: np.ndarray) -> float:
        data = data[~np.isnan(data)]
        check_len(data)
        var = self.rfvar3(data, np.ones((len(data), 1)))
        return np.sqrt(np.var(var["u"]))

    @staticmethod
    def rfvar3(ydata: np.ndarray, xdata: np.ndarray) -> dict:
        if len(ydata.shape) == 1:
            ydata = ydata.reshape((len(ydata), 1))
        t, n_var = ydata.shape
        nox = xdata is None
        t2, nx = xdata.shape
        smpl = np.array([np.arange(1, t)]).T

        t_sample = smpl.shape[0]
        x = np.zeros((t_sample, n_var))
        for i in range(t_sample):
            x[i] = ydata[smpl[i] - np.arange(1, 1 + 1)].T
        x = np.concatenate((x, xdata[np.arange(1, t)]), axis=1)
        y = ydata[np.arange(1, t)]

        vl, d_diag, vr = np.linalg.svd(x)
        di = np.array([d_diag]).T
        vr = vr.T.conj()

        dfx = np.sum(di > 100 * 2.2204e-16)
        singularity = x.shape[1] - dfx
        di = np.divide(1, di[0:dfx])

        vl = vl[:, 0:dfx]
        vr = vr[:, 0:dfx]
        b = np.dot(vl.T, y)

        b = np.dot(np.dot(vr, np.diag(di.flatten())), b)
        u = y - np.dot(x, b)
        xxi = np.dot(vr, np.diag(di.flatten()))
        xxi = np.dot(xxi, xxi.T)

        b = b.reshape((n_var + nx, n_var))
        by = b[0:n_var]
        by = by.reshape((n_var, 1, n_var))
        by = np.transpose(by, axes=(2, 0, 1))
        if nox:
            bx = []
        else:
            bx = b[n_var + np.arange(nx), :].T
        return {
            "By": by,
            "Bx": bx[0],
            "u": u,
            "xxi": xxi,
            "singularity": singularity,
        }
