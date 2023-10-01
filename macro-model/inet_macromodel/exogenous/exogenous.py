import numpy as np
import pandas as pd

from inet_macromodel.exogenous.exogenous_ts import create_exogenous_timeseries


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
        start_ind = np.where(self.log_inflation.index == str(initial_year) + "-1")[0][0]
        self.log_inflation_before = self.log_inflation.iloc[0:start_ind]
        self.log_inflation_during = self.log_inflation.iloc[start_ind : start_ind + t_max - offset]

        # Cutting off growth
        start_ind = np.where(self.sectoral_growth.index == str(initial_year) + "-1")[0][0]
        self.sectoral_growth_before = self.sectoral_growth.iloc[0:start_ind]
        self.sectoral_growth_during = self.sectoral_growth.iloc[start_ind : start_ind + t_max - offset]

        # Cutting off unemployment rate
        start_ind = np.where(self.unemployment_rate.index == str(initial_year) + "-1")[0][0]
        self.unemployment_rate_before = self.unemployment_rate.iloc[0:start_ind]
        self.unemployment_rate_during = self.unemployment_rate.iloc[start_ind : start_ind + t_max - offset]

        # Cutting off vacancy rate
        start_ind = np.where(self.vacancy_rate.index == str(initial_year) + "-1")[0][0]
        self.vacancy_rate_before = self.vacancy_rate.iloc[0:start_ind]
        self.vacancy_rate_during = self.vacancy_rate.iloc[start_ind : start_ind + t_max - offset]

        # Cutting off the housing price index
        start_ind = np.where(self.house_price_index.index == str(initial_year) + "-1")[0][0]
        self.house_price_index_before = self.house_price_index.iloc[0:start_ind]
        self.house_price_index_during = self.house_price_index.iloc[start_ind : start_ind + t_max - offset]

        # Cutting off total firm deposits and debt
        start_ind = np.where(self.total_firm_deposits_and_debt.index == str(initial_year) + "-1")[0][0]
        self.total_firm_deposits_and_debt_before = self.total_firm_deposits_and_debt.iloc[0:start_ind]
        self.total_firm_deposits_and_debt_during = self.total_firm_deposits_and_debt.iloc[
            start_ind : start_ind + t_max - offset
        ]

        # Cutting off industry-level data
        if len(self.iot_industry_data) > 0:
            new_cols_2 = []
            for i in range(18):
                new_cols_2 += [i] * int(len(self.iot_industry_data.columns) / 18)
            new_cols = [
                (list(self.iot_industry_data.columns.get_level_values(0))[i], new_cols_2[i])
                for i in range(len(new_cols_2))
            ]
            self.iot_industry_data.columns = pd.MultiIndex.from_tuples(new_cols)
            start_ind = np.where(self.iot_industry_data.index == str(initial_year) + "-1")[0][0]
            self.iot_industry_data_before = self.iot_industry_data.iloc[0:start_ind]
            self.iot_industry_data_during = self.iot_industry_data.iloc[start_ind + 11 : start_ind + 22 + t_max]
        else:
            self.iot_industry_data_before = pd.DataFrame(
                {
                    "Output in LCU": [],
                    "Household Consumption in LCU": [],
                    "Government Consumption in LCU": [],
                    "Imports in LCU": [],
                    "Exports in LCU": [],
                }
            )
            for c in all_country_names:
                self.iot_industry_data_before["Imports in LCU from " + c] = []
                self.iot_industry_data_before["Exports in LCU to " + c] = []
            self.iot_industry_data_during = pd.DataFrame(
                {
                    "Output in LCU": [],
                    "Household Consumption in LCU": [],
                    "Government Consumption in LCU": [],
                    "Imports in LCU": [],
                    "Exports in LCU": [],
                }
            )
            for c in all_country_names:
                self.iot_industry_data_during["Imports in LCU from " + c] = []
                self.iot_industry_data_during["Exports in LCU to " + c] = []

        # Cutting-off exchange rates
        self.exchange_rates_data = self.exchange_rates_data.T
        self.exchange_rates_data.index = [ind + "-01" for ind in self.exchange_rates_data.index]
        self.exchange_rates_data.index = pd.to_datetime(self.exchange_rates_data.index)
        self.exchange_rates_data_before = self.exchange_rates_data.loc[
            self.exchange_rates_data.index < pd.Timestamp(initial_year, 1, 1)
        ]
        self.exchange_rates_data_during = self.exchange_rates_data.loc[
            self.exchange_rates_data.index >= pd.Timestamp(initial_year, 1, 1)
        ]

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
        """
        if len(self.iot_industry_data) > 0:
            for col in self.iot_industry_data_before.columns:
                self.iot_industry_data_before.loc[:, col] = self.iot_industry_data_before[col].fillna(
                    self.iot_industry_data[col].mean()
                )
        """

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

        # Compile historic data
        self.compiled_historic_data = self.compile_historic_data()

    def compile_historic_data(self) -> pd.DataFrame:
        # Stuff
        cpi_before = self.log_inflation_before[["Real CPI Inflation"]]
        cpi_before.index = pd.MultiIndex.from_product([cpi_before.index, [0]], names=["Date", "Industry"])
        ppi_before = self.log_inflation_before[["Real PPI Inflation"]]
        ppi_before.index = pd.MultiIndex.from_product([ppi_before.index, [0]], names=["Date", "Industry"])
        sec_growth_before = pd.DataFrame(self.sectoral_growth_before.stack())
        sec_growth_before.index.names = ["Date", "Industry"]
        sec_growth_before.columns = ["Sectoral Growth"]
        ur_before = self.unemployment_rate_before.copy()
        ur_before.index = pd.MultiIndex.from_product([ur_before.index, [0]], names=["Date", "Industry"])
        vr_before = self.vacancy_rate_before.copy()
        vr_before.index = pd.MultiIndex.from_product([vr_before.index, [0]], names=["Date", "Industry"])
        hpi_before = self.house_price_index_before.copy()
        hpi_before.index = pd.MultiIndex.from_product([hpi_before.index, [0]], names=["Date", "Industry"])
        total_debt_deposits = self.total_firm_deposits_and_debt_before.copy()
        total_debt_deposits.index = pd.MultiIndex.from_product(
            [total_debt_deposits.index, [0]], names=["Date", "Industry"]
        )

        # Exchange rates
        exchange_rate_before_monthly_data = []
        exchange_rate_before_monthly_index = []
        for ind in self.exchange_rates_data_before.index:
            for m in range(1, 13):
                exchange_rate_before_monthly_data.append(self.exchange_rates_data_before.loc[ind].values[0])
                exchange_rate_before_monthly_index.append(str(ind.year) + "-" + str(m))
        exchange_rate_before = pd.DataFrame(
            data=exchange_rate_before_monthly_data, index=exchange_rate_before_monthly_index
        )
        exchange_rate_before.index = pd.MultiIndex.from_product(
            [exchange_rate_before.index, [0]], names=["Date", "Industry"]
        )
        exchange_rate_before.columns = ["Exchange Rate"]

        # Put it together
        data_except_iot = pd.concat(
            [
                cpi_before,
                ppi_before,
                sec_growth_before,
                ur_before,
                vr_before,
                hpi_before,
                total_debt_deposits,
                exchange_rate_before,
            ],
            axis=1,
        )
        new_index = list(data_except_iot.index)
        for i in range(len(new_index)):
            if "-10" in new_index[i][0] or "-11" in new_index[i][0] or "-12" in new_index[i][0]:  # noqa
                continue
            for j in range(1, 10):
                new_index[i] = (new_index[i][0].replace("-" + str(j), "-0" + str(j)), new_index[i][1])  # noqa
        data_except_iot.index = pd.MultiIndex.from_tuples(new_index, names=["Date", "Industry"])

        # IOT data
        iot_industry_data_before = self.iot_industry_data_before.stack()
        new_index = list(iot_industry_data_before.index)
        for i in range(len(new_index)):
            new_index[i] = (str(new_index[i][0])[0:-12], new_index[i][1])  # noqa
        iot_industry_data_before.index = pd.MultiIndex.from_tuples(new_index, names=["Date", "Industry"])

        # Put it together
        complete_data = pd.concat(
            [
                data_except_iot,
                iot_industry_data_before,
            ],
            axis=1,
        )
        complete_data.columns = [field.replace(" ", "_").upper() for field in complete_data.columns]
        dates = list(dict.fromkeys(complete_data.index.get_level_values(0)))
        historic_exogenous_data = complete_data.reindex(pd.MultiIndex.from_product([dates, range(18)]))
        historic_exogenous_data = historic_exogenous_data.loc[
            historic_exogenous_data.index.get_level_values(0).str[0:4].astype(int) >= 2009
        ]

        return historic_exogenous_data
