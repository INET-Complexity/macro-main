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
    """
    A class representing synthetic central banks.

    The central bank data is stored in a pandas DataFrame with a single column:
        - Policy Rate: The policy rate.

    Attributes:
        country_name (str): The name of the country.
        year (int): The year.
        central_bank_data (pd.DataFrame): The central bank data, containing the policy rate and possibly more data.

    Methods:
        __init__(country_name, year, central_bank_data): Initializes a SyntheticDefaultCentralBanks instance.
        from_readers(country_name, year, readers): Initializes a SyntheticDefaultCentralBanks instance from readers.
    """

    def __init__(
        self,
        country_name: str,
        year: int,
        central_bank_data: pd.DataFrame,
    ):
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
        """
        Initializes a SyntheticCentralBank object using data from DataReaders,
        in particular storing the central bank policy rate.

        Args:
            country_name (str): The name of the country.
            year (int): The year.
            readers (DataReaders): An instance of DataReaders.
            exogenous_data (ExogenousCountryData): An instance of ExogenousCountryData.
            central_bank_configuration (CentralBankDataConfiguration): An instance of CentralBankDataConfiguration.

        Returns:
            SyntheticCentralBank: An instance of SyntheticCentralBank.
        """
        policy_rates = readers.policy_rates.get_policy_rates(country_name)
        inflation = exogenous_data.inflation["PPI Inflation"]
        growth = exogenous_data.national_accounts["Gross Output (Growth)"]
        targeted_inflation_rate = central_bank_configuration.inflation_target

        merged = pd.merge_asof(policy_rates, inflation, left_index=True, right_index=True)
        merged = pd.merge_asof(merged, growth, left_index=True, right_index=True)

        merged = merged.loc[merged.index < pd.to_datetime(f"{year}-Q{quarter}")]

        # drop rows with NaN values
        merged = merged.dropna()

        excess_inflation = merged["PPI Inflation"].values - targeted_inflation_rate
        growth_values = merged["Gross Output (Growth)"].values

        exog = np.array(list(zip(excess_inflation, growth_values)))
        order = {i: [1] for i in range(exog.shape[1])}

        # Fit a linear regression model
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

        central_bank_data = {
            "targeted_inflation_rate": [targeted_inflation_rate],
            "rho": [res.params[1]],
            "r_star": [res.params[0] / (1 - res.params[1]) - targeted_inflation_rate],
            "xi_pi": [res.params[2] / (1 - res.params[1])],
            "xi_gamma": [res.params[3] / (1 - res.params[1])],
        }

        # TODO: the xi_pi factor is wrong

        central_bank_data["policy_rate"] = [
            max(
                0.0,
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
