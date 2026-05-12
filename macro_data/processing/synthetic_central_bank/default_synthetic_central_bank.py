"""Default implementation for preprocessing synthetic central bank data.

This module provides a concrete implementation for preprocessing central bank data
that will be used to initialize behavioral models. Key preprocessing includes:

1. Data Collection and Processing:
   - Historical policy rate collection
   - Inflation and growth data aggregation
   - Parameter estimation from historical data

2. Taylor Rule Parameter Estimation:
   - Interest rate smoothing calculation
   - Response coefficients estimation
   - Natural rate computation

3. Data Organization:
   - Time series alignment
   - Missing data handling
   - Data validation

Note:
    This module is NOT used for simulating central bank behavior. It preprocesses
    data that will be used to initialize behavioral models in the simulation package.
    The actual policy decisions and rate setting are implemented elsewhere.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.api import ARDL

from macro_data.configuration.dataconfiguration import CentralBankDataConfiguration
from macro_data.processing.synthetic_central_bank.synthetic_central_bank import (
    SyntheticCentralBank,
)
from macro_data.readers.default_readers import DataReaders
from macro_data.readers.exogenous_data import ExogenousCountryData
from macro_data.util.frequency import periods_per_year


class DefaultSyntheticCentralBank(SyntheticCentralBank):
    """Default implementation for preprocessing central bank data.

    This class preprocesses and organizes central bank data by estimating Taylor rule
    parameters from historical data. These parameters will be used to initialize
    behavioral models, but this class does NOT implement any behavioral logic.

    The preprocessed data DataFrame contains:
        - policy_rate: Historical/initial policy rate
        - targeted_inflation_rate: Reference inflation target
        - rho: Estimated interest rate smoothing parameter
        - r_star: Estimated natural real interest rate
        - xi_pi: Estimated inflation response coefficient
        - xi_gamma: Estimated growth response coefficient

    The parameter estimation uses the form:
    r_t = ρr_{t-1} + (1-ρ)[r* + π* + ξ_π(π_t - π*) + ξ_γγ_t]
    to extract parameters from historical data, where:
        r_t: historical policy rate
        ρ: smoothing parameter
        r*: natural rate
        π*: inflation target
        π_t: historical inflation
        γ_t: historical growth

    Note:
        This is a data container class. The actual monetary policy implementation
        occurs in the simulation package, which uses these preprocessed parameters.

    Attributes:
        country_name (str): Country identifier for data collection
        year (int): Reference year for preprocessing
        central_bank_data (pd.DataFrame): Preprocessed parameter data
    """

    def __init__(
        self,
        country_name: str,
        year: int,
        central_bank_data: pd.DataFrame,
    ):
        """Initialize the central bank data container.

        Args:
            country_name (str): Country identifier for data collection
            year (int): Reference year for preprocessing
            central_bank_data (pd.DataFrame): Initial data containing:
                - Historical rates
                - Estimated parameters
                - Target values
        """
        super().__init__(
            country_name,
            year,
            central_bank_data,
        )

    @classmethod
    def from_readers(
        cls,
        country_name: str,
        year: int,
        quarter: int,
        readers: DataReaders,
        exogenous_data: ExogenousCountryData,
        central_bank_configuration: CentralBankDataConfiguration,
        time_unit: int,
    ):
        """Create a preprocessed central bank data container using historical data.

        This method preprocesses historical data to estimate parameters:
        1. Collects and aligns historical time series data
        2. Estimates Taylor rule parameters using ARDL models
        3. Organizes parameters and initial values for model initialization

        The preprocessing steps:
        1. Merge policy rates with macro indicators
        2. Estimate parameters via ARDL regression
        3. Transform parameters to structural form
        4. Compute initial policy rate using estimated parameters

        Args:
            country_name (str): Country to preprocess data for
            year (int): Reference year for preprocessing
            quarter (int): Reference quarter (1-4)
            readers (DataReaders): Data source readers
            exogenous_data (ExogenousCountryData): External economic data
            central_bank_configuration (CentralBankDataConfiguration): Configuration settings
            time_unit (int): Simulation period length in months, used to build
                one-year CPI YoY inflation and the output-gap trend.

        Returns:
            DefaultSyntheticCentralBank: Container with preprocessed parameters
        """
        policy_rates = readers.policy_rates.get_policy_rates(country_name)
        inflation = exogenous_data.inflation["PPI Inflation"]
        growth = exogenous_data.national_accounts["Gross Output (Growth)"]
        targeted_inflation_rate = central_bank_configuration.inflation_target

        # Merge and prepare data for estimation
        merged = pd.merge_asof(policy_rates, inflation, left_index=True, right_index=True)
        merged = pd.merge_asof(merged, growth, left_index=True, right_index=True)
        merged = merged.loc[merged.index < pd.to_datetime(f"{year}-Q{quarter}")]
        merged = merged.dropna()

        # Prepare variables for estimation
        excess_inflation = merged["PPI Inflation"].values - targeted_inflation_rate
        growth_values = merged["Gross Output (Growth)"].values
        exog = np.array(list(zip(excess_inflation, growth_values)))
        order = {i: [1] for i in range(exog.shape[1])}

        # Estimate policy rule
        model = ARDL(
            endog=merged["Policy Rate"].values.astype(float),
            lags=1,
            exog=exog.astype(float),
            order=order,
            causal=False,
            trend="c",
            seasonal=False,
        )
        res = model.fit()

        # Extract and transform parameters
        inflation_response = res.params[2] / (1 - res.params[1])
        growth_response = res.params[3] / (1 - res.params[1])
        smooth_taylor_params = cls._estimate_smooth_taylor_rule(
            policy_rates=policy_rates,
            exogenous_data=exogenous_data,
            targeted_inflation_rate=targeted_inflation_rate,
            year=year,
            quarter=quarter,
            time_unit=time_unit,
        )
        central_bank_data = {
            "targeted_inflation_rate": [targeted_inflation_rate],
            "rho": [res.params[1]],  # Interest rate smoothing
            "r_star": [res.params[0] / (1 - res.params[1]) - targeted_inflation_rate],  # Natural rate
            "xi_pi": [inflation_response],  # Poledna inflation response
            "xi_gamma": [growth_response],  # Poledna growth response
            "smooth_rho": [smooth_taylor_params["rho"]],
            "smooth_r_star": [smooth_taylor_params["r_star"]],
            "phi_pi": [smooth_taylor_params["phi_pi"]],
            "phi_q": [smooth_taylor_params["phi_q"]],
            "smooth_policy_rate": [smooth_taylor_params["policy_rate"]],
        }

        # TODO: the xi_pi factor is wrong

        # Keep the legacy Poledna initialization anchored to the last observed
        # policy rate before the simulation start. This preserves a clear
        # interpretation for the stored value even though the Poledna rule itself
        # remains on its legacy convention.
        central_bank_data["policy_rate"] = [max(0.0, float(merged["Policy Rate"].values[-1]))]

        central_bank_data = pd.DataFrame(central_bank_data)
        return cls(country_name, year, central_bank_data)

    @classmethod
    def _compute_cpi_yoy_inflation(cls, cpi_inflation: pd.Series, time_unit: int) -> pd.Series:
        """Build a YoY CPI inflation series from per-period CPI inflation."""
        periods_per_year_value = periods_per_year(time_unit)
        compounded = (1.0 + cpi_inflation.astype(float)).rolling(periods_per_year_value).apply(np.prod, raw=True)
        return compounded - 1.0

    @classmethod
    def _compute_output_gap(cls, real_gross_output: pd.Series, time_unit: int) -> pd.Series:
        """Build an output-gap series using a one-year EWMA trend."""
        periods_per_year_value = periods_per_year(time_unit)
        safe_real_output = real_gross_output.astype(float).clip(lower=1e-12)
        potential_output = safe_real_output.ewm(span=periods_per_year_value, adjust=False).mean().clip(lower=1e-12)
        return np.log(safe_real_output) - np.log(potential_output)

    @classmethod
    def _estimate_smooth_taylor_rule(
        cls,
        policy_rates: pd.DataFrame,
        exogenous_data: ExogenousCountryData,
        targeted_inflation_rate: float,
        year: int,
        quarter: int,
        time_unit: int,
    ) -> dict[str, float]:
        """Estimate SmoothTaylorRule parameters on annual policy rates.

        The estimated regression is:

            i_t = c + rho * i_{t-1} + beta_pi * (pi_t^yoy - pi*) + beta_q * q_t

        where `pi_t^yoy` is YoY CPI inflation and `q_t` is the output gap. The
        structural Taylor-rule coefficients are recovered by dividing the reduced-form
        loadings by `(1 - rho)`.
        """
        cpi_yoy_inflation = cls._compute_cpi_yoy_inflation(exogenous_data.inflation["CPI Inflation"], time_unit)
        output_gap = cls._compute_output_gap(exogenous_data.national_accounts["Real Gross Output (Value)"], time_unit)

        merged = pd.merge_asof(
            policy_rates.sort_index(),
            cpi_yoy_inflation.rename("CPI YoY Inflation").sort_index(),
            left_index=True,
            right_index=True,
        )
        merged = pd.merge_asof(
            merged,
            output_gap.rename("Output Gap").sort_index(),
            left_index=True,
            right_index=True,
        )
        merged = merged.loc[merged.index < pd.to_datetime(f"{year}-Q{quarter}")]
        merged = merged.dropna()

        cpi_gap = merged["CPI YoY Inflation"].values.astype(float) - targeted_inflation_rate
        exog = np.array(list(zip(cpi_gap, merged["Output Gap"].values.astype(float))))
        order = {i: 0 for i in range(exog.shape[1])}

        model = ARDL(
            endog=merged["Policy Rate"].values.astype(float),
            lags=1,
            exog=exog,
            order=order,
            causal=False,
            trend="c",
            seasonal=False,
        )
        res = model.fit()

        rho = float(res.params[1])
        phi_pi = float(res.params[2] / (1 - rho))
        phi_q = float(res.params[3] / (1 - rho))
        r_star = float(res.params[0] / (1 - rho) - targeted_inflation_rate)
        periods_per_year_value = periods_per_year(time_unit)

        return {
            "rho": rho,
            "r_star": r_star,
            "phi_pi": phi_pi,
            "phi_q": phi_q,
            # SmoothTaylorRule stores rates in per-period units inside the simulation.
            "policy_rate": max(0.0, float(merged["Policy Rate"].values[-1]) / periods_per_year_value),
        }
