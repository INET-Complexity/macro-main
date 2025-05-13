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

4. Vector Autoregressive (VAR) Forecasting:
   - Multivariate time series modeling
   - Joint variable dynamics
   - Multiple simulation paths

The module supports both deterministic and stochastic forecasts, with options
to suppress noise terms for scenario analysis. Each forecaster implements a
common interface while providing specific prediction mechanics.
"""

from abc import ABC, abstractmethod
from typing import Literal, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller


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


class VARForecaster(Forecaster):
    """Vector Autoregressive (VAR) forecaster.

    Implements multivariate time series forecasting using VAR models,
    supporting multiple variables and simulation paths.

    Attributes:
        var_lags (int): Number of lags in the VAR model
        country (str): Country code for the forecast
        model (VAR): Fitted VAR model
    """

    def __init__(self, var_lags: int = 2, country: str = None):
        """Initialize VAR forecaster.

        Args:
            var_lags (int, optional): Number of lags in the VAR model. Defaults to 2.
            country (str, optional): Country code. Defaults to None.
        """
        self.var_lags = var_lags
        self.country = country
        self.model = None
        self.data = None

    def fit(self, data: pd.DataFrame) -> 'VARForecaster':
        """Fit the VAR model to the data.

        Args:
            data (pd.DataFrame): Time series data with variables as columns

        Returns:
            VARForecaster: Self for method chaining
        """
        self.data = data
        self.model = VAR(data)
        self.model = self.model.fit(self.var_lags)
        return self

    def forecast(self, data: np.ndarray, t: int = 1, assume_zero_noise: bool = False) -> np.ndarray:
        """Generate VAR forecasts.

        Args:
            data (np.ndarray): Historical time series (unused if model is fitted)
            t (int, optional): Forecast horizon. Defaults to 1.
            assume_zero_noise (bool, optional): Whether to suppress stochastic
                components. Defaults to False.

        Returns:
            np.ndarray: Forecasted values
        """
        if self.model is None:
            raise ValueError("Model must be fitted before forecasting")
            
        if assume_zero_noise:
            # Use deterministic forecast
            forecast = self.model.forecast(self.data.values[-self.var_lags:], steps=t)
        else:
            # Use stochastic simulation
            forecast = self.model.simulate_var(steps=t, nsimulations=1)[0]
            
        return forecast

    def simulate(self, horizon: int = 20, num_paths: int = 500) -> pd.DataFrame:
        """Generate multiple simulation paths.

        Args:
            horizon (int, optional): Number of periods to simulate. Defaults to 20.
            num_paths (int, optional): Number of simulation paths. Defaults to 500.

        Returns:
            pd.DataFrame: Simulated values with columns for variable, period, and value
        """
        if self.model is None:
            raise ValueError("Model must be fitted before simulation")
            
        sims = self.model.simulate_var(steps=horizon, nsimulations=num_paths)
        
        results = []
        for path in range(num_paths):
            for period in range(horizon):
                for var_idx, var_name in enumerate(self.data.columns):
                    results.append({
                        'country': self.country,
                        'variable': var_name,
                        'period': period,
                        'value': sims[path, period, var_idx]
                    })
                    
        return pd.DataFrame(results)

    @staticmethod
    def check_stationarity(series: pd.Series, alpha: float = 0.05) -> bool:
        """Check if a time series is stationary using Augmented Dickey-Fuller test.

        Args:
            series (pd.Series): Time series to test
            alpha (float, optional): Significance level. Defaults to 0.05.

        Returns:
            bool: True if series is stationary, False otherwise
        """
        result = adfuller(series.dropna())
        return result[1] < alpha

    @staticmethod
    def prepare_data(data_wrapper, country: str) -> pd.DataFrame:
        """Prepare time series data for VAR modeling.

        Args:
            data_wrapper: DataWrapper instance containing country data
            country (str): Country code

        Returns:
            pd.DataFrame: Prepared time series data
        """
        country_data = data_wrapper.synthetic_countries[country]
        
        data = pd.DataFrame({
            'real_gdp_growth': country_data.exogenous_data.national_accounts['Real GDP (Growth)'],
            'inflation': country_data.exogenous_data.inflation['PPI Inflation'],
            'real_consumption': country_data.exogenous_data.national_accounts['Real Household Consumption (Growth)'],
            'real_investment': country_data.exogenous_data.national_accounts['Gross Fixed Capital Formation (Growth)']
        })
        
        # Take first differences of real consumption and real investment
        data['real_consumption'] = data['real_consumption'].diff()
        data['real_investment'] = data['real_investment'].diff()
        
        # Handle missing values
        data = data.ffill().bfill()
        data = data.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        # Ensure sufficient data points
        if len(data) < 10:
            raise ValueError(f"Insufficient data points for country {country}. Need at least 10 observations.")
        
        # Check stationarity and difference if necessary
        for col in data.columns:
            if not VARForecaster.check_stationarity(data[col]):
                print(f"Warning: {col} is not stationary for {country}, differencing...")
                data[col] = data[col].diff().dropna()
                if not VARForecaster.check_stationarity(data[col]):
                    print(f"Warning: Differenced {col} is still not stationary for {country}")
        
        # Drop NaN values and check multicollinearity
        data = data.dropna()
        corr_matrix = data.corr()
        if (corr_matrix.abs() > 0.95).any().any():
            print(f"Warning: High correlation detected in data for country {country}")
            print("Correlation matrix:")
            print(corr_matrix)
            
        # Standardize data
        data = (data - data.mean()) / data.std()
        
        return data

    @staticmethod
    def run_multiple_countries(data_wrapper, countries: list[str], horizon: int = 20, num_paths: int = 500, var_lags: int = 2) -> pd.DataFrame:
        """Run VAR model for multiple countries and combine results.

        Args:
            data_wrapper: DataWrapper instance containing country data
            countries (list[str]): List of country codes to analyze
            horizon (int, optional): Number of periods to simulate. Defaults to 20.
            num_paths (int, optional): Number of simulation paths. Defaults to 500.
            var_lags (int, optional): Number of lags in the VAR model. Defaults to 2.

        Returns:
            pd.DataFrame: Combined simulation results for all countries
        """
        all_results = []
        
        for country in countries:
            try:
                data = VARForecaster.prepare_data(data_wrapper, country)
                forecaster = VARForecaster(var_lags=var_lags, country=country)
                forecaster.fit(data)
                results = forecaster.simulate(horizon=horizon, num_paths=num_paths)
                all_results.append(results)
                
            except Exception as e:
                print(f"Error processing country {country}: {str(e)}")
                continue
            
        if not all_results:
            raise ValueError("No countries were successfully processed")
            
        return pd.concat(all_results, ignore_index=True)
