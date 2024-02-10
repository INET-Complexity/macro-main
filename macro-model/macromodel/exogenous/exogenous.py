import h5py
import pandas as pd
from macro_data import SyntheticCountry

from macromodel.exchange_rates.exchange_rates import ExchangeRates
from macromodel.exogenous.exogenous_ts import create_exogenous_timeseries


class Exogenous:
    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        initial_year: int,
        t_max: int,
        log_inflation: pd.DataFrame,
        sectoral_growth: pd.DataFrame,
        unemployment_rate: pd.DataFrame,
        vacancy_rate: pd.DataFrame,
        house_price_index: pd.DataFrame,
        total_firm_deposits_and_debt: pd.DataFrame,
        iot_industry_data: pd.DataFrame,
        exchange_rates_data: pd.DataFrame,
    ):
        self.country_name = country_name
        self.log_inflation = log_inflation
        self.sectoral_growth = sectoral_growth
        self.unemployment_rate = unemployment_rate
        self.vacancy_rate = vacancy_rate
        self.house_price_index = house_price_index
        self.total_firm_deposits_and_debt = total_firm_deposits_and_debt
        self.iot_industry_data = iot_industry_data
        self.exchange_rates_data = exchange_rates_data

        offset = 0

        # Cutting off inflation

        initial_year = pd.to_datetime(f"{initial_year}-01-01")

        year_before = initial_year - pd.DateOffset(days=1)

        end_date = initial_year + pd.DateOffset(months=t_max)

        self.log_inflation_before = self.log_inflation.loc[:year_before]
        self.log_inflation_during = self.log_inflation.loc[initial_year:end_date]

        # Cutting off growth
        self.sectoral_growth_before = self.sectoral_growth.loc[:year_before]
        self.sectoral_growth_during = self.sectoral_growth.loc[initial_year:end_date]

        # Cutting off unemployment rate
        self.unemployment_rate_before = self.unemployment_rate.loc[:year_before]
        self.unemployment_rate_during = self.unemployment_rate.loc[initial_year:end_date]

        # Cutting off vacancy rate
        self.vacancy_rate_before = self.vacancy_rate.loc[:year_before]
        self.vacancy_rate_during = self.vacancy_rate.loc[initial_year:end_date]

        # Cutting off the housing price index
        self.house_price_index_before = self.house_price_index.loc[:year_before]
        self.house_price_index_during = self.house_price_index.loc[initial_year:end_date]

        # Cutting off total firm deposits and debt
        self.total_firm_deposits_and_debt_before = self.total_firm_deposits_and_debt.loc[:year_before]
        self.total_firm_deposits_and_debt_during = self.total_firm_deposits_and_debt.loc[initial_year:end_date]

        # Cutting off industry-level data

        self.iot_industry_data_before = self.iot_industry_data.loc[:year_before]
        self.iot_industry_data_during = self.iot_industry_data.loc[initial_year:end_date]

        # Cutting-off exchange rates
        self.exchange_rates_data = self.exchange_rates_data.T
        self.exchange_rates_data.index = pd.to_datetime(self.exchange_rates_data.index, format="%Y")

        self.exchange_rates_data_before = self.exchange_rates_data.loc[:year_before]
        self.exchange_rates_data_during = self.exchange_rates_data.loc[initial_year:end_date]

        # Impute missing values for inflation
        self.log_inflation_before.loc[:, "Real CPI Inflation"] = self.log_inflation_before["Real CPI Inflation"].fillna(
            self.log_inflation["Real CPI Inflation"].mean()
        )
        self.log_inflation_before.loc[:, "Real PPI Inflation"] = self.log_inflation_before["Real PPI Inflation"].fillna(
            self.log_inflation["Real PPI Inflation"].mean()
        )

        # Impute missing values for growth
        for g in self.sectoral_growth_before.columns:
            self.sectoral_growth_before.loc[:, g] = self.sectoral_growth_before[g].fillna(
                self.sectoral_growth[g].mean()
            )
        self.sectoral_growth_before.columns = range(len(self.sectoral_growth_before.columns))

        # Impute missing values for the house price index
        self.house_price_index_before.loc[:, "Real House Price Index Growth"] = self.house_price_index_before[
            "Real House Price Index Growth"
        ].fillna(self.house_price_index["Real House Price Index Growth"].mean())
        self.house_price_index_before.loc[:, "Nominal House Price Index Growth"] = self.house_price_index_before[
            "Nominal House Price Index Growth"
        ].fillna(self.house_price_index["Nominal House Price Index Growth"].mean())

        # Impute missing values for the industry-level data

        # if len(self.iot_industry_data) > 0:
        #     for col in self.iot_industry_data_before.columns:
        #         self.iot_industry_data_before.loc[:, col] = self.iot_industry_data_before[col].fillna(
        #             self.iot_industry_data[col].mean()
        #         )

        # Put it all together in a time series object
        self.ts = create_exogenous_timeseries(
            country_name=country_name,
            all_country_names=all_country_names,
            log_inflation_during=self.log_inflation_during,
            sectoral_growth_during=self.sectoral_growth_during,
            unemployment_rate_during=self.unemployment_rate_during,
            vacancy_rate_during=self.vacancy_rate_during,
            house_price_index_during=self.house_price_index_during,
            total_firm_deposits_and_debt_during=self.total_firm_deposits_and_debt_during,
            iot_industry_data_during=self.iot_industry_data_during,
            exchange_rates_data_during=self.exchange_rates_data_during,
        )

        # # Compile historic data
        # self.compiled_historic_data = self.compile_historic_data()

    @classmethod
    def from_pickled_agent(
        cls,
        synthetic_country: SyntheticCountry,
        exchange_rates: ExchangeRates,
        country_name: str,
        all_country_names: list[str],
        initial_year: int,
        t_max: int,
    ):
        exogenous_data = synthetic_country.exogenous_data
        return cls(
            country_name,
            all_country_names,
            initial_year,
            t_max,
            log_inflation=exogenous_data.log_inflation,
            sectoral_growth=exogenous_data.sectoral_growth,
            unemployment_rate=exogenous_data.unemployment_rate,
            vacancy_rate=exogenous_data.vacancy_rate,
            house_price_index=exogenous_data.house_price_index,
            total_firm_deposits_and_debt=exogenous_data.total_firm_deposits_and_debt,
            iot_industry_data=exogenous_data.iot_industry_data,
            exchange_rates_data=exchange_rates.historic_exchange_rate_data.loc[[country_name]],
        )

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
