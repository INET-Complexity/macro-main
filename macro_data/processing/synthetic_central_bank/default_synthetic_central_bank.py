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
        central_bank_data = {
            "targeted_inflation_rate": [targeted_inflation_rate],
            "rho": [res.params[1]],  # Interest rate smoothing
            "r_star": [res.params[0] / (1 - res.params[1]) - targeted_inflation_rate],  # Natural rate
            "xi_pi": [res.params[2] / (1 - res.params[1])],  # Inflation response
            "xi_gamma": [res.params[3] / (1 - res.params[1])],  # Growth response
        }

        # TODO: the xi_pi factor is wrong

        # Compute current policy rate with zero lower bound
        central_bank_data["policy_rate"] = [
            max(
                0.0,  # Zero lower bound
                central_bank_data["rho"][0] * merged["Policy Rate"].values[-1]
                + (1 - central_bank_data["rho"][0])
                * (
                    central_bank_data["r_star"][0]
                    + targeted_inflation_rate
                    + central_bank_data["xi_pi"][0] * (excess_inflation[-1] - targeted_inflation_rate)
                    + central_bank_data["xi_gamma"][0] * growth_values[-1]
                ),
            )
        ]

        central_bank_data = pd.DataFrame(central_bank_data)
        return cls(country_name, year, central_bank_data)
