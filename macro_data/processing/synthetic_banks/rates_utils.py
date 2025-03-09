"""Utilities for managing and calculating bank interest rates.

This module provides tools for analyzing and predicting interest rates in the banking
system. It implements:

1. Rate Data Processing:
   - Merging different types of rates (firm, household, mortgage)
   - Handling time series data with proper alignment
   - Computing rate differentials and changes

2. Rate Model Fitting:
   - ARDL (Autoregressive Distributed Lag) model implementation
   - Separate models for different loan types
   - Optimal lag selection using AIC

3. Default Rate Handling:
   - Fallback mechanisms for insufficient data
   - Policy rate based defaults
   - Consistent handling across rate types

The module supports both EU and non-EU countries and handles various data quality
scenarios through robust fallback mechanisms.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.ardl import ARDL


def rates_dataframe(
    firm_rates: pd.Series,
    household_consumption_rates: pd.Series,
    household_mortgage_rates: pd.Series,
    inflation: pd.DataFrame,
    npl_rates: pd.DataFrame,
    policy_rates: pd.DataFrame,
    year: int,
    quarter: int,
) -> pd.DataFrame:
    """Create a consolidated DataFrame of various interest rates and related metrics.

    This function:
    1. Merges different types of rates into a single DataFrame
    2. Aligns timestamps using asof merge for irregular data
    3. Computes rate differentials
    4. Filters data up to specified year/quarter

    Args:
        firm_rates (pd.Series): Interest rates for firm loans
        household_consumption_rates (pd.Series): Rates for consumer loans
        household_mortgage_rates (pd.Series): Mortgage rates
        inflation (pd.DataFrame): Inflation data
        npl_rates (pd.DataFrame): Non-performing loan rates
        policy_rates (pd.DataFrame): Central bank policy rates
        year (int): Target year for filtering
        quarter (int): Target quarter for filtering

    Returns:
        pd.DataFrame: Consolidated DataFrame containing:
            - Original rates (firm, consumer, mortgage)
            - Rate differentials
            - NPL rates
            - Policy rates
            - Inflation data
    """
    # merge all the rates
    df = pd.DataFrame(
        {
            "firm_rates": firm_rates,
            "household_consumption_rates": household_consumption_rates,
            "household_mortgage_rates": household_mortgage_rates,
        }
    )
    npl_rates.columns = ["npl"]
    # use mergeasof to add npl rates
    df = pd.merge_asof(df, npl_rates, left_index=True, right_index=True)
    # use mergeasof to add policy rates
    df = pd.merge_asof(df, policy_rates, left_index=True, right_index=True)
    # use mergeasof to add inflation rates
    df = pd.merge_asof(df, inflation, left_index=True, right_index=True)
    # fillnans with the last value
    df[["npl", "Policy Rate"]] = df[["npl", "Policy Rate"]].ffill()
    # drop rows with nans
    df = df.dropna()
    df["firm_rates_diff"] = df["firm_rates"].diff()
    df["household_consumption_rates_diff"] = df["household_consumption_rates"].diff()
    df["household_mortgage_rates_diff"] = df["household_mortgage_rates"].diff()
    df["inflation_diff"] = df["PPI Inflation"].diff()
    # get the good indices
    mask = df.index < pd.Timestamp(f"{year}Q{quarter}")
    df = df.loc[mask]
    df.dropna()
    return df


def default_rate_values(policy_rates: pd.DataFrame) -> tuple[float, float, float]:
    """Get default values for rate calculations when insufficient data is available.

    Args:
        policy_rates (pd.DataFrame): Central bank policy rates

    Returns:
        tuple[float, float, float]: Default values for:
            - Rate passthrough (1.0)
            - Error correction term (-1.0)
            - Latest policy rate
    """
    return 1.0, -1.0, policy_rates["Policy Rate"].values[-1]


def fit_mortgage_models(df: pd.DataFrame, n_lags: int, min_size: int = 12) -> tuple[float, float, float]:
    """Fit ARDL models for mortgage rates.

    This function:
    1. Checks for sufficient data
    2. Fits ARDL models with varying lags
    3. Computes passthrough and error correction terms
    4. Calculates predicted mortgage rates

    Args:
        df (pd.DataFrame): Rate data including mortgage rates
        n_lags (int): Maximum number of lags to consider
        min_size (int, optional): Minimum required observations. Defaults to 12.

    Returns:
        tuple[float, float, float]:
            - Mortgage rate passthrough
            - Error correction term
            - Predicted mortgage rate
    """
    if len(df) < min_size:
        return default_rate_values(df[["Policy Rate"]])

    exog = df[["household_mortgage_rates", "Policy Rate", "npl"]].values
    rates_diff = df["household_mortgage_rates_diff"].values
    best_lag, best_model = fit_models(exog, n_lags, rates_diff)
    if best_model.params[best_lag + 1] == 0.0:
        pt = 0
    else:
        pt = -best_model.params[best_lag + 2] / best_model.params[best_lag + 1]
    household_mortgages_pt = pt
    household_mortgages_ect = best_model.params[best_lag + 1]
    household_mortgages_rates = df["household_mortgage_rates"].values[-1] + household_mortgages_ect * (
        df["household_mortgage_rates"].values[-1] - household_mortgages_pt * df["Policy Rate"].values[-1]
    )

    return household_mortgages_pt, household_mortgages_ect, household_mortgages_rates


def fit_household_models(df: pd.DataFrame, n_lags: int, min_size: int = 12) -> tuple[float, float, float]:
    """Fit ARDL models for household consumption loan rates.

    This function:
    1. Checks for sufficient data
    2. Fits ARDL models with varying lags
    3. Computes passthrough and error correction terms
    4. Calculates predicted consumer loan rates

    Args:
        df (pd.DataFrame): Rate data including consumer loan rates
        n_lags (int): Maximum number of lags to consider
        min_size (int, optional): Minimum required observations. Defaults to 12.

    Returns:
        tuple[float, float, float]:
            - Consumer rate passthrough
            - Error correction term
            - Predicted consumer loan rate
    """
    if len(df) < min_size:
        return default_rate_values(df[["Policy Rate"]])
    exog = df[["household_consumption_rates", "Policy Rate", "npl"]].values
    rates_diff = df["household_consumption_rates_diff"].values
    best_lag, best_model = fit_models(exog, n_lags, rates_diff)
    if best_model.params[best_lag + 1] == 0.0:
        pt = 0
    else:
        pt = -best_model.params[best_lag + 2] / best_model.params[best_lag + 1]
    household_consumption_pt = pt
    household_consumption_ect = best_model.params[best_lag + 1]
    household_consumption_rates = df["household_consumption_rates"].values[-1] + household_consumption_ect * (
        df["household_consumption_rates"].values[-1] - household_consumption_pt * df["Policy Rate"].values[-1]
    )
    return household_consumption_pt, household_consumption_ect, household_consumption_rates


def fit_firm_models(df: pd.DataFrame, n_lags: int, min_size: int = 12) -> tuple[float, float, float]:
    """Fit ARDL models for firm loan rates.

    This function:
    1. Checks for sufficient data
    2. Fits ARDL models with varying lags
    3. Computes passthrough and error correction terms
    4. Calculates predicted firm loan rates

    Args:
        df (pd.DataFrame): Rate data including firm loan rates
        n_lags (int): Maximum number of lags to consider
        min_size (int, optional): Minimum required observations. Defaults to 12.

    Returns:
        tuple[float, float, float]:
            - Firm rate passthrough
            - Error correction term
            - Predicted firm loan rate
    """
    if len(df) < min_size:
        return default_rate_values(df[["Policy Rate"]])
    exog = df[["firm_rates", "Policy Rate", "npl"]].values
    rates_diff = df["firm_rates_diff"].values
    best_lag, best_model = fit_models(exog, n_lags, rates_diff)
    if best_model.params[best_lag + 1] == 0.0:
        pt = 0
    else:
        pt = -best_model.params[best_lag + 2] / best_model.params[best_lag + 1]
    firm_pt = pt
    firm_ect = best_model.params[best_lag + 1]
    firm_rates = df["firm_rates"].values[-1] + firm_ect * (
        df["firm_rates"].values[-1] - firm_pt * df["Policy Rate"].values[-1]
    )
    return firm_pt, firm_ect, firm_rates


def fit_models(exog: np.ndarray, n_lags: int, rates_diff: np.ndarray) -> tuple[int, ARDL]:
    """Fit ARDL models with different lag specifications and select the best one.

    This function:
    1. Creates ARDL models with different lag structures
    2. Fits each model to the data
    3. Selects the best model based on AIC

    Args:
        exog (np.ndarray): Exogenous variables (rates and NPL data)
        n_lags (int): Maximum number of lags to consider
        rates_diff (np.ndarray): Rate differentials for model fitting

    Returns:
        tuple[int, ARDL]:
            - Best lag order
            - Fitted ARDL model with best lag structure
    """
    # Pick the optimal number of lags based on the AIC
    order = {j: [1] for j in range(exog.shape[1])}
    models = [
        ARDL(
            endog=rates_diff[1:],
            lags=lag_endo,
            exog=exog[1:],
            order=order,
            causal=False,
            trend="c",
            seasonal=False,
        ).fit()
        for lag_endo in range(n_lags)
    ]
    best_lag, best_model = min(enumerate(models), key=lambda x: x[1].aic)
    return best_lag, best_model
