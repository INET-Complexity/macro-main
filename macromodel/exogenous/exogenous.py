"""Exogenous data management module.

This module manages external economic data that influences the model but is not
determined within it. It serves as the bridge between historical/external data
and the dynamic simulation, providing:

1. Historical Data Management:
   - Pre-simulation economic indicators
   - Calibration period data
   - Time series organization and splitting

2. Economic Indicators:
   - Inflation rates (CPI, PPI)
   - National accounts (GDP components)
   - Labor market metrics (unemployment, vacancy)
   - House price indices
   - Exchange rates

3. Simulation Support:
   - Initial conditions for model start
   - Calibration data for parameter estimation
   - Reference points for model validation
   - External constraints and benchmarks

The module ensures proper temporal alignment and organization of external data,
critical for model initialization and ongoing calibration.
"""

import h5py
import numpy as np
import pandas as pd

from macro_data import SyntheticCountry
from macromodel.exchange_rates.exchange_rates import ExchangeRates
from macromodel.exogenous.exogenous_ts import create_exogenous_timeseries


class Exogenous:
    """External economic data manager.

    This class handles the organization and provision of external economic data
    that influences but is not determined by the model. It splits historical
    data into pre-simulation and simulation periods, maintaining proper temporal
    alignment and data organization.

    Attributes:
        country_name (str): Associated country identifier
        inflation (pd.DataFrame): CPI and PPI inflation data
        national_accounts (pd.DataFrame): GDP and components data
        unemployment_rate (pd.DataFrame): Labor market unemployment data
        vacancy_rate (pd.DataFrame): Labor market vacancy data
        house_price_index (pd.DataFrame): Property market indices
        exchange_rates_data (pd.DataFrame): Currency exchange rates
        inflation_before (pd.DataFrame): Pre-simulation inflation data
        inflation_during (pd.DataFrame): Simulation period inflation data
        national_accounts_before (pd.DataFrame): Pre-simulation national accounts
        national_accounts_during (pd.DataFrame): Simulation period accounts
        unemployment_rate_before (pd.DataFrame): Pre-simulation unemployment
        unemployment_rate_during (pd.DataFrame): Simulation period unemployment
        vacancy_rate_before (pd.DataFrame): Pre-simulation vacancy rates
        vacancy_rate_during (pd.DataFrame): Simulation period vacancy rates
        house_price_index_before (pd.DataFrame): Pre-simulation house prices
        house_price_index_during (pd.DataFrame): Simulation period house prices
        exchange_rates_data_before (pd.DataFrame): Pre-simulation exchange rates
        exchange_rates_data_during (pd.DataFrame): Simulation period rates
        ts (TimeSeries): Organized time series of all exogenous data
        compiled_historic_data (pd.DataFrame): Combined pre-simulation data
    """

    def __init__(
        self,
        country_name: str,
        initial_year: int,
        initial_quarter: int,
        t_max: int,
        inflation: pd.DataFrame,
        national_accounts: pd.DataFrame,
        unemployment_rate: pd.DataFrame,
        vacancy_rate: pd.DataFrame,
        house_price_index: pd.DataFrame,
        exchange_rates_data: pd.DataFrame,
    ):
        """Initialize exogenous data manager.

        Args:
            country_name (str): Country identifier
            initial_year (int): Start year for simulation
            initial_quarter (int): Start quarter for simulation
            t_max (int): Maximum simulation periods
            inflation (pd.DataFrame): Inflation rate data
            national_accounts (pd.DataFrame): National accounts data
            unemployment_rate (pd.DataFrame): Unemployment rate data
            vacancy_rate (pd.DataFrame): Vacancy rate data
            house_price_index (pd.DataFrame): House price indices
            exchange_rates_data (pd.DataFrame): Exchange rate data
        """
        self.country_name = country_name
        self.inflation = inflation
        self.national_accounts = national_accounts
        self.unemployment_rate = unemployment_rate
        self.vacancy_rate = vacancy_rate
        self.house_price_index = house_price_index
        self.exchange_rates_data = exchange_rates_data

        offset = 0

        # Split data into before/during simulation periods
        start_ind = np.where(self.inflation.index == str(initial_year) + "-Q" + str(initial_quarter))[0][0]
        self.inflation_before = self.inflation.iloc[0:start_ind]
        self.inflation_during = self.inflation.iloc[start_ind : start_ind + t_max - offset]

        if len(self.national_accounts) > 0:
            self.national_accounts_before = self.national_accounts.loc[
                self.national_accounts.index < pd.Timestamp(initial_year, 3 * initial_quarter - 2, 1)
            ]
            self.national_accounts_during = self.national_accounts.loc[
                self.national_accounts.index >= pd.Timestamp(initial_year, 3 * initial_quarter - 2, 1)
            ]
        else:
            self.national_accounts_before = pd.DataFrame()
            self.national_accounts_during = pd.DataFrame()

        start_ind = np.where(self.unemployment_rate.index == str(initial_year) + "-Q" + str(initial_quarter))[0][0]
        self.unemployment_rate_before = self.unemployment_rate.iloc[0:start_ind]
        self.unemployment_rate_during = self.unemployment_rate.iloc[start_ind : start_ind + t_max - offset]

        start_ind = np.where(self.vacancy_rate.index == str(initial_year) + "-Q" + str(initial_quarter))[0][0]
        self.vacancy_rate_before = self.vacancy_rate.iloc[0:start_ind]
        self.vacancy_rate_during = self.vacancy_rate.iloc[start_ind : start_ind + t_max - offset]

        start_ind = np.where(self.house_price_index.index == str(initial_year) + "-Q" + str(initial_quarter))[0][0]
        self.house_price_index_before = self.house_price_index.iloc[0:start_ind]
        self.house_price_index_during = self.house_price_index.iloc[start_ind : start_ind + t_max - offset]

        # Process exchange rates
        self.exchange_rates_data = self.exchange_rates_data.T
        self.exchange_rates_data = self.exchange_rates_data.loc[:, country_name]
        self.exchange_rates_data.index = [ind for ind in self.exchange_rates_data.index]
        self.exchange_rates_data.index = pd.PeriodIndex(self.exchange_rates_data.index, freq="Q").to_timestamp()
        self.exchange_rates_data.columns = ["Exchange Rate"]
        self.exchange_rates_data_before = self.exchange_rates_data.loc[
            self.exchange_rates_data.index < pd.Timestamp(initial_year, 3 * initial_quarter - 2, 1)
        ]
        self.exchange_rates_data_during = self.exchange_rates_data.loc[
            self.exchange_rates_data.index >= pd.Timestamp(initial_year, 3 * initial_quarter - 2, 1)
        ]

        # Create time series and compile historic data
        self.ts = create_exogenous_timeseries(
            inflation_during=self.inflation_during,
            national_accounts_during=self.national_accounts_during,
            unemployment_rate_during=self.unemployment_rate_during,
            vacancy_rate_during=self.vacancy_rate_during,
            house_price_index_during=self.house_price_index_during,
            exchange_rates_data_during=self.exchange_rates_data_during,
        )

        self.compiled_historic_data = pd.concat(
            [
                self.inflation_before,
                self.national_accounts_before,
                self.unemployment_rate_before,
                self.vacancy_rate_before,
                self.house_price_index_before,
                self.exchange_rates_data_before,
            ],
            axis=1,
        )
        self.compiled_historic_data.columns = [
            field.replace(" ", "_").upper() for field in self.compiled_historic_data.columns
        ]

    @classmethod
    def from_pickled_agent(
        cls,
        synthetic_country: SyntheticCountry,
        exchange_rates: ExchangeRates,
        country_name: str,
        initial_year: int,
        t_max: int,
    ):
        """Create instance from synthetic country data.

        Factory method that constructs an Exogenous instance using
        preprocessed synthetic country data.

        Args:
            synthetic_country (SyntheticCountry): Preprocessed country data
            exchange_rates (ExchangeRates): Exchange rate dynamics
            country_name (str): Country identifier
            initial_year (int): Start year
            t_max (int): Maximum periods

        Returns:
            Exogenous: Initialized exogenous data manager
        """
        exogenous_data = synthetic_country.exogenous_data
        return cls(
            country_name=country_name,
            exchange_rates_data=exchange_rates.historic_exchange_rate_data,
            inflation=exogenous_data.inflation,
            national_accounts=exogenous_data.national_accounts,
            unemployment_rate=exogenous_data.labour_stats[["Unemployment Rate (Value)"]],
            vacancy_rate=exogenous_data.labour_stats[["Vacancy Rate (Value)"]],
            house_price_index=exogenous_data.house_price_index,
            initial_year=initial_year,
            initial_quarter=1,
            t_max=t_max,
        )

    def reset(self) -> None:
        """Reset time series to initial state."""
        self.ts.reset()

    def save_to_h5(self, group: h5py.Group) -> None:
        """Save exogenous data to HDF5.

        Args:
            group (h5py.Group): HDF5 group to save to
        """
        self.ts.write_to_h5("exogenous", group)

    # def compile_historic_data(self) -> pd.DataFrame:
    #     # Stuff
    #     cpi_before = self.log_inflation_before[["Real CPI Inflation"]]
    #     cpi_before.index = pd.MultiIndex.from_product([cpi_before.index, [0]], names=["Date", "Industry"])
    #     ppi_before = self.log_inflation_before[["Real PPI Inflation"]]
    #     ppi_before.index = pd.MultiIndex.from_product([ppi_before.index, [0]], names=["Date", "Industry"])
    #     sec_growth_before = pd.DataFrame(self.sectoral_growth_before.stack())
    #     sec_growth_before.index.names = ["Date", "Industry"]
    #     sec_growth_before.columns = ["Sectoral Growth"]
    #     ur_before = self.unemployment_rate_before.copy()
    #     ur_before.index = pd.MultiIndex.from_product([ur_before.index, [0]], names=["Date", "Industry"])
    #     vr_before = self.vacancy_rate_before.copy()
    #     vr_before.index = pd.MultiIndex.from_product([vr_before.index, [0]], names=["Date", "Industry"])
    #     hpi_before = self.house_price_index_before.copy()
    #     hpi_before.index = pd.MultiIndex.from_product([hpi_before.index, [0]], names=["Date", "Industry"])
    #     total_debt_deposits = self.total_firm_deposits_and_debt_before.copy()
    #     total_debt_deposits.index = pd.MultiIndex.from_product(
    #         [total_debt_deposits.index, [0]], names=["Date", "Industry"]
    #     )
    #
    #     # Exchange rates
    #     exchange_rate_before = self.exchange_rates_data_before.copy()
    #
    #     exchange_rate_before.index = pd.MultiIndex.from_product(
    #         [exchange_rate_before.index, [0]], names=["Date", "Industry"]
    #     )
    #     exchange_rate_before.columns = ["Exchange Rate"]
    #
    #     # Put it together
    #     data_except_iot = pd.concat(
    #         [
    #             cpi_before,
    #             ppi_before,
    #             sec_growth_before,
    #             ur_before,
    #             vr_before,
    #             hpi_before,
    #             total_debt_deposits,
    #             exchange_rate_before,
    #         ],
    #         axis=1,
    #     )
    #     new_index = list(data_except_iot.index)
    #     for i in range(len(new_index)):
    #         if "-10" in new_index[i][0] or "-11" in new_index[i][0] or "-12" in new_index[i][0]:  # noqa
    #             continue
    #         for j in range(1, 10):
    #             new_index[i] = (new_index[i][0].replace("-" + str(j), "-0" + str(j)), new_index[i][1])  # noqa
    #     data_except_iot.index = pd.MultiIndex.from_tuples(new_index, names=["Date", "Industry"])
    #
    #     # IOT data
    #     iot_industry_data_before = self.iot_industry_data_before.stack()
    #     new_index = list(iot_industry_data_before.index)
    #     for i in range(len(new_index)):
    #         new_index[i] = (str(new_index[i][0])[0:-12], new_index[i][1])  # noqa
    #     iot_industry_data_before.index = pd.MultiIndex.from_tuples(new_index, names=["Date", "Industry"])
    #
    #     # Put it together
    #     complete_data = pd.concat(
    #         [
    #             data_except_iot,
    #             iot_industry_data_before,
    #         ],
    #         axis=1,
    #     )
    #     complete_data.columns = [field.replace(" ", "_").upper() for field in complete_data.columns]
    #     dates = list(dict.fromkeys(complete_data.index.get_level_values(0)))
    #     historic_exogenous_data = complete_data.reindex(pd.MultiIndex.from_product([dates, range(18)]))
    #     historic_exogenous_data = historic_exogenous_data.loc[
    #         historic_exogenous_data.index.get_level_values(0).str[0:4].astype(int) >= 2009
    #     ]
    #
    #     return historic_exogenous_data
