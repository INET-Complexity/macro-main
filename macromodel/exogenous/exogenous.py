import h5py
import numpy as np
import pandas as pd

from macro_data import SyntheticCountry
from macromodel.exchange_rates.exchange_rates import ExchangeRates
from macromodel.exogenous.exogenous_ts import create_exogenous_timeseries


class Exogenous:
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
        self.country_name = country_name
        self.inflation = inflation
        self.national_accounts = national_accounts
        self.unemployment_rate = unemployment_rate
        self.vacancy_rate = vacancy_rate
        self.house_price_index = house_price_index
        self.exchange_rates_data = exchange_rates_data

        offset = 0

        # Cutting off inflation
        start_ind = np.where(self.inflation.index == str(initial_year) + "-Q" + str(initial_quarter))[0][0]
        self.inflation_before = self.inflation.iloc[0:start_ind]
        self.inflation_during = self.inflation.iloc[start_ind : start_ind + t_max - offset]

        # Cutting off GDP decomp
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

        # Cutting off unemployment rate
        start_ind = np.where(self.unemployment_rate.index == str(initial_year) + "-Q" + str(initial_quarter))[0][0]
        self.unemployment_rate_before = self.unemployment_rate.iloc[0:start_ind]
        self.unemployment_rate_during = self.unemployment_rate.iloc[start_ind : start_ind + t_max - offset]

        # Cutting off vacancy rate
        start_ind = np.where(self.vacancy_rate.index == str(initial_year) + "-Q" + str(initial_quarter))[0][0]
        self.vacancy_rate_before = self.vacancy_rate.iloc[0:start_ind]
        self.vacancy_rate_during = self.vacancy_rate.iloc[start_ind : start_ind + t_max - offset]

        # Cutting off the housing price index
        start_ind = np.where(self.house_price_index.index == str(initial_year) + "-Q" + str(initial_quarter))[0][0]
        self.house_price_index_before = self.house_price_index.iloc[0:start_ind]
        self.house_price_index_during = self.house_price_index.iloc[start_ind : start_ind + t_max - offset]

        # Cutting-off exchange rates
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

        # Put it all together in a time series object
        self.ts = create_exogenous_timeseries(
            inflation_during=self.inflation_during,
            national_accounts_during=self.national_accounts_during,
            unemployment_rate_during=self.unemployment_rate_during,
            vacancy_rate_during=self.vacancy_rate_during,
            house_price_index_during=self.house_price_index_during,
            exchange_rates_data_during=self.exchange_rates_data_during,
        )

        # Compile historic data
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
        self.ts.reset()

    def save_to_h5(self, group: h5py.Group):
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
