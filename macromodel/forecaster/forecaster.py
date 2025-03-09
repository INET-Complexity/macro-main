"""Economic time series forecasting module.

This module provides a collection of forecasting methods used throughout the
macroeconomic model for predicting various economic variables. It implements
several forecasting strategies:

1. Constant Forecasting:
   - Fixed value predictions
   - Useful for steady-state assumptions
   - Simple baseline forecasts

2. OLS (Trend) Forecasting:
   - Linear trend extrapolation
   - Time series regression
   - Growth rate projections

3. Autoregressive Forecasting:
   - AR(p) model implementations
   - Both manual and statsmodels-based
   - Handles stochastic components

The module supports both deterministic and stochastic forecasts, with options
to suppress noise terms for scenario analysis. Each forecaster implements a
common interface while providing specific prediction mechanics.
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg


def check_len(data: np.ndarray) -> None:
    """Check if array has sufficient length for forecasting.

    Args:
        data (np.ndarray): Time series data to check

    Raises:
        ValueError: If array length is less than 3
    """
    if len(data) < 3:
        raise ValueError("Array is too small to be forecasted", data)


class Forecaster(ABC):
    """Abstract base class for time series forecasting.

    Provides common interface for all forecasting implementations,
    ensuring consistent prediction methods across different approaches.
    """

    @abstractmethod
    def forecast(self, data: np.ndarray, t: int = 1, assume_zero_noise: bool = False) -> float | np.ndarray:
        """Generate forecasts from time series data.

        Args:
            data (np.ndarray): Historical time series to forecast from
            t (int, optional): Number of periods ahead to forecast. Defaults to 1.
            assume_zero_noise (bool, optional): Whether to suppress stochastic
                components. Defaults to False.

        Returns:
            float | np.ndarray: Forecasted values
        """
        pass


class ConstantForecaster(Forecaster):
    """Fixed value forecaster.

    Implements simple constant value forecasting, useful for steady-state
    assumptions or baseline scenarios.

    Attributes:
        value (float): Fixed value to forecast
    """

    def __init__(self, value: float) -> None:
        """Initialize constant forecaster.

        Args:
            value (float): Fixed value to use in forecasts
        """
        self.value = value

    def forecast(self, data: np.ndarray, t: int = 1, assume_zero_noise: bool = False) -> np.ndarray:
        """Generate constant value forecasts.

        Args:
            data (np.ndarray): Historical data (unused)
            t (int, optional): Forecast periods (unused). Defaults to 1.
            assume_zero_noise (bool, optional): Noise flag (unused). Defaults to False.

        Returns:
            np.ndarray: Array containing the fixed forecast value
        """
        return np.array([self.value])


class OLSForecaster(Forecaster):
    """Linear trend forecaster.

    Implements trend-based forecasting using OLS regression,
    extrapolating linear trends in the data.
    """

    def forecast(self, data: np.ndarray, t: int = 1, assume_zero_noise: bool = False) -> np.ndarray:
        """Generate trend-based forecasts.

        Fits linear trend to historical data and extrapolates forward.

        Args:
            data (np.ndarray): Historical time series
            t (int, optional): Forecast periods (unused). Defaults to 1.
            assume_zero_noise (bool, optional): Noise flag (unused). Defaults to False.

        Returns:
            np.ndarray: Array containing the trend forecast
        """
        check_len(data)
        x = sm.add_constant(np.arange(len(data)))
        model = sm.OLS(data, x).fit()
        prediction = model.params[0] + model.params[1] * len(data)
        return np.array([prediction])


class ImplementedAutoregForecaster(Forecaster):
    """Statsmodels-based autoregressive forecaster.

    Implements AR(p) forecasting using statsmodels' AutoReg,
    supporting various trend specifications.

    Attributes:
        lags (int): Number of autoregressive lags
        trend (str): Trend specification ('n', 'c', 't', or 'ct')
    """

    def __init__(self, lags: int, trend: Literal["n", "c", "t", "ct"] = "t"):
        """Initialize autoregressive forecaster.

        Args:
            lags (int): Number of AR lags to use
            trend (str, optional): Trend specification:
                'n': no trend
                'c': constant
                't': linear trend
                'ct': constant and trend
                Defaults to "t".
        """
        self.lags = lags
        self.trend = trend

    def forecast(self, data: np.ndarray, t: int = 1, assume_zero_noise: bool = False) -> float | np.ndarray:
        """Generate autoregressive forecasts.

        Args:
            data (np.ndarray): Historical time series
            t (int, optional): Forecast horizon. Defaults to 1.
            assume_zero_noise (bool, optional): Noise flag (unused). Defaults to False.

        Returns:
            float | np.ndarray: Forecasted values
        """
        check_len(data)
        forecast = AutoReg(data, lags=[self.lags], trend=self.trend, old_names=False).fit().forecast(steps=t)
        return forecast


class ManualAutoregForecaster(Forecaster):
    """Custom autoregressive forecaster implementation.

    Implements AR(1) forecasting with manual computation and optional
    stochastic components. Supports both deterministic and stochastic
    forecasts.
    """

    def forecast(
        self,
        data: np.ndarray,
        t: int = 1,
        assume_zero_noise: bool = False,
    ) -> np.ndarray:
        """Generate manual AR(1) forecasts.

        Implements AR(1) forecasting with optional noise terms.

        Args:
            data (np.ndarray): Historical time series
            t (int, optional): Forecast horizon. Defaults to 1.
            assume_zero_noise (bool, optional): Whether to suppress
                stochastic components. Defaults to False.

        Returns:
            np.ndarray: Forecasted values
        """
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
        """Calculate variance of the AR model residuals.

        Args:
            data (np.ndarray): Historical time series

        Returns:
            float: Standard deviation of residuals
        """
        data = data[~np.isnan(data)]
        check_len(data)
        var = self.rfvar3(data, np.ones((len(data), 1)))
        return np.sqrt(np.var(var["u"]))

    @staticmethod
    def rfvar3(ydata: np.ndarray, xdata: np.ndarray) -> dict:
        """Fit reduced form VAR model using SVD.

        Implements VAR estimation using singular value decomposition
        for numerical stability.

        Args:
            ydata (np.ndarray): Dependent variable data
            xdata (np.ndarray): Independent variable data

        Returns:
            dict: Fitted model parameters including:
                - By: AR coefficients
                - Bx: Constant terms
                - u: Residuals
                - xxi: Covariance matrix
                - singularity: Rank deficiency
        """
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
