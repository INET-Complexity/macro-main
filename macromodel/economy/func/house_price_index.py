"""House Price Index (HPI) forecasting module.

This module provides various approaches for forecasting house price growth
using different statistical and econometric methods. It implements several
forecasting strategies:

1. Constant Growth:
   - Fixed HPI growth rate predictions
   - Useful for baseline scenarios

2. OLS-based Growth:
   - Linear regression approach
   - Captures basic price dynamics

3. Autoregressive Growth:
   - Both implemented and manual AR models
   - Accounts for price momentum
   - Supports different lag structures

Each forecasting method can be bounded by minimum and maximum growth rates
to ensure economically sensible predictions and prevent extreme values.
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


class HPIForecasting(ABC):
    """Abstract base class for house price index forecasting.

    Provides a common interface for different HPI forecasting approaches.
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

    def forecast_hpi_growth(
        self,
        historic_hpi: np.ndarray,
        min_hpi_growth: Optional[float] = None,
        max_hpi_growth: Optional[float] = None,
        t: int = 1,
        assume_zero_noise: bool = False,
    ) -> float:
        """Generate HPI growth forecasts based on historical data.

        Args:
            historic_hpi (np.ndarray): Past HPI growth rates
            min_hpi_growth (Optional[float], optional): Lower bound. Defaults to None.
            max_hpi_growth (Optional[float], optional): Upper bound. Defaults to None.
            t (int, optional): Forecast horizon. Defaults to 1.
            assume_zero_noise (bool, optional): Suppress random variation. Defaults to False.

        Returns:
            float: Forecasted HPI growth rate
        """
        forecast = self.forecaster.forecast(
            historic_hpi,
            t,
            assume_zero_noise=assume_zero_noise,
        )
        if min_hpi_growth is not None:
            forecast = np.maximum(min_hpi_growth, forecast)
        if max_hpi_growth is not None:
            forecast = np.maximum(max_hpi_growth, forecast)
        return forecast


class HPIForecastingConstant(HPIForecasting):
    """Constant HPI growth forecasting implementation.

    Predicts a fixed house price growth rate regardless of history.
    Useful for baseline scenarios and policy analysis.

    Attributes:
        forecaster (ConstantForecaster): Fixed-value forecasting model
    """

    def __init__(self, value: float, *args, **kwargs):
        """Initialize constant forecaster.

        Args:
            value (float): Fixed HPI growth rate to predict
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__()
        self.forecaster = ConstantForecaster(value=value)


class HPIForecastingOLS(HPIForecasting):
    """OLS-based HPI growth forecasting implementation.

    Uses linear regression to predict house price growth.
    Captures basic linear relationships in price dynamics.

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


class HPIImplementedForecastingAutoReg(HPIForecasting):
    """Implemented autoregressive HPI growth forecasting.

    Uses a pre-implemented AR model for house price prediction.
    Accounts for price momentum through lag structure.

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


class HPIManualForecastingAutoReg(HPIForecasting):
    """Manual autoregressive HPI growth forecasting.

    Uses a manually implemented AR(1) model for house price prediction.
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
