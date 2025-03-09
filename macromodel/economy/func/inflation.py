"""Inflation forecasting module for economic simulations.

This module provides various approaches for forecasting inflation rates using
different statistical and econometric methods. It implements several forecasting
strategies:

1. Constant Forecasting:
   - Fixed inflation rate predictions
   - Useful for scenario analysis and baseline comparisons

2. OLS-based Forecasting:
   - Linear regression approach
   - Captures basic inflation dynamics

3. Autoregressive Forecasting:
   - Both implemented and manual AR models
   - Accounts for inflation persistence
   - Supports different lag structures

4. Exogenous Forecasting:
   - Uses externally provided inflation paths
   - Suitable for policy analysis and stress testing

Each forecasting method can be bounded by minimum and maximum values to
ensure economically sensible predictions.
"""

from abc import ABC
from typing import Optional

import numpy as np

from macromodel.forecaster.forecaster import (
    ConstantForecaster,
    ImplementedAutoregForecaster,
    ManualAutoregForecaster,
    OLSForecaster,
)


class InflationForecasting(ABC):
    """Abstract base class for inflation forecasting methods.

    Provides a common interface for different inflation forecasting approaches.
    Supports bounded forecasts and noise control in predictions.

    Attributes:
        forecaster: Underlying forecasting model implementation
    """

    def __init__(self, *args, **kwargs):
        """Initialize forecasting instance.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        self.forecaster = None

    def forecast_inflation(
        self,
        historic_inflation: np.ndarray,
        exogenous_inflation: np.ndarray,
        current_time: int,
        min_inflation: Optional[float] = None,
        max_inflation: Optional[float] = None,
        t: int = 1,
        assume_zero_noise: bool = False,
    ) -> float | np.ndarray:
        """Generate inflation forecasts based on historical data.

        Args:
            historic_inflation (np.ndarray): Past inflation rates
            exogenous_inflation (np.ndarray): External inflation factors
            current_time (int): Current period index
            min_inflation (Optional[float], optional): Lower bound. Defaults to None.
            max_inflation (Optional[float], optional): Upper bound. Defaults to None.
            t (int, optional): Forecast horizon. Defaults to 1.
            assume_zero_noise (bool, optional): Suppress random variation. Defaults to False.

        Returns:
            float | np.ndarray: Forecasted inflation rate(s)
        """
        forecast = self.forecaster.forecast(
            historic_inflation,
            t,
            assume_zero_noise=assume_zero_noise,
        )
        if min_inflation is not None:
            forecast = np.maximum(min_inflation, forecast)
        if max_inflation is not None:
            forecast = np.maximum(max_inflation, forecast)
        return forecast


class InflationForecastingConstant(InflationForecasting):
    """Constant inflation forecasting implementation.

    Predicts a fixed inflation rate regardless of historical data.
    Useful for baseline scenarios and policy analysis.

    Attributes:
        forecaster (ConstantForecaster): Fixed-value forecasting model
    """

    def __init__(self, value: float, *args, **kwargs):
        """Initialize constant forecaster.

        Args:
            value (float): Fixed inflation rate to predict
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__()
        self.forecaster = ConstantForecaster(value=value)


class InflationForecastingOLS(InflationForecasting):
    """OLS-based inflation forecasting implementation.

    Uses linear regression to predict inflation based on historical data.
    Captures basic linear relationships in inflation dynamics.

    Attributes:
        forecaster (OLSForecaster): Linear regression forecasting model
    """

    def __init__(self, *args, **kwargs):
        """Initialize OLS forecaster.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__()
        self.forecaster = OLSForecaster()


class InflationImplementedForecastingAutoReg(InflationForecasting):
    """Implemented autoregressive inflation forecasting.

    Uses a pre-implemented AR model for inflation prediction.
    Accounts for inflation persistence through lag structure.

    Attributes:
        forecaster (ImplementedAutoregForecaster): AR model implementation
    """

    def __init__(self, lags: int, *args, **kwargs):
        """Initialize implemented AR forecaster.

        Args:
            lags (int): Number of autoregressive lags
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__()
        self.forecaster = ImplementedAutoregForecaster(lags)


class InflationManualForecastingAutoReg(InflationForecasting):
    """Manual autoregressive inflation forecasting.

    Uses a manually implemented AR(1) model for inflation prediction.
    Provides direct control over the autoregressive process.

    Attributes:
        forecaster (ManualAutoregForecaster): Manual AR model implementation
    """

    def __init__(self, lags: int, *args, **kwargs):
        """Initialize manual AR forecaster.

        Args:
            lags (int): Must be 1 (AR(1) only)
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Raises:
            AssertionError: If lags != 1
        """
        assert lags == 1
        super().__init__()
        self.forecaster = ManualAutoregForecaster()


class ExogenousInflationForecasting(InflationForecasting):
    """Exogenous inflation forecasting implementation.

    Uses externally provided inflation paths rather than generating forecasts.
    Suitable for scenario analysis and policy simulations.
    """

    def __init__(self, lags: int, *args, **kwargs):
        """Initialize exogenous forecaster.

        Args:
            lags (int): Not used but kept for interface consistency
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__()
        self.forecaster = None

    def forecast_inflation(
        self,
        historic_inflation: np.ndarray,
        exogenous_inflation: np.ndarray,
        current_time: int,
        min_inflation: Optional[float] = None,
        max_inflation: Optional[float] = None,
        t: int = 1,
        assume_zero_noise: bool = False,
    ) -> float | np.ndarray:
        """Return exogenous inflation value for current period.

        Args:
            historic_inflation (np.ndarray): Not used
            exogenous_inflation (np.ndarray): External inflation path
            current_time (int): Current period index
            min_inflation (Optional[float], optional): Not used. Defaults to None.
            max_inflation (Optional[float], optional): Not used. Defaults to None.
            t (int, optional): Not used. Defaults to 1.
            assume_zero_noise (bool, optional): Not used. Defaults to False.

        Returns:
            float | np.ndarray: Current period's exogenous inflation
        """
        return np.array([exogenous_inflation[current_time]])
