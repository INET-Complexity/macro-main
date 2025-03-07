"""Forecasting module for economic time series analysis.

This module provides a collection of forecasting tools and implementations
for economic time series analysis. It includes:

1. Base forecasting interface (Forecaster)
2. Simple forecasting methods (ConstantForecaster)
3. Trend-based forecasting (OLSForecaster)
4. Autoregressive implementations:
   - Statsmodels-based AR(p) (ImplementedAutoregForecaster)
   - Custom AR(1) with noise control (ManualAutoregForecaster)

The module supports both deterministic and stochastic forecasting approaches,
with configurable parameters for trend specification, lag selection, and
noise components.
"""

from .forecaster import (
    Forecaster,
    ConstantForecaster,
    OLSForecaster,
    ImplementedAutoregForecaster,
    ManualAutoregForecaster,
)

__all__ = [
    "Forecaster",
    "ConstantForecaster",
    "OLSForecaster",
    "ImplementedAutoregForecaster",
    "ManualAutoregForecaster",
]
