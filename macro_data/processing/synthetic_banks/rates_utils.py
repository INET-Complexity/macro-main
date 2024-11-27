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
):
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


def default_rate_values(policy_rates: pd.DataFrame):
    return 1.0, -1.0, policy_rates["Policy Rate"].values[-1]


def fit_mortgage_models(df: pd.DataFrame, n_lags: int, min_size: int = 12):
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


def fit_household_models(df: pd.DataFrame, n_lags: int, min_size: int = 12):
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


def fit_firm_models(df: pd.DataFrame, n_lags: int, min_size: int = 12):
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


def fit_models(exog: np.ndarray, n_lags: int, rates_diff: np.ndarray):
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
