import pandas as pd

from inet_macromodel.timeseries import TimeSeries


def create_exogenous_timeseries(
    country_name: str,
    all_country_names: list[str],
    log_inflation_during: pd.DataFrame,
    sectoral_growth_during: pd.DataFrame,
    unemployment_rate_during: pd.DataFrame,
    vacancy_rate_during: pd.DataFrame,
    house_price_index_during: pd.DataFrame,
    total_firm_deposits_and_debt_during: pd.DataFrame,
    iot_industry_data_during: pd.DataFrame,
    exchange_rates_data_during: pd.DataFrame,
) -> TimeSeries:
    exog_ts = TimeSeries(
        cpi_inflation=[log_inflation_during["Real CPI Inflation"].values[0]],
        ppi_inflation=[log_inflation_during["Real PPI Inflation"].values[0]],
        unemployment_rate=[unemployment_rate_during.values[0]],
        vacancy_rate=[vacancy_rate_during.values[0]],
        real_house_price_index_growth=[house_price_index_during["Real House Price Index Growth"].values[0]],
        nominal_house_price_index_growth=[house_price_index_during["Nominal House Price Index Growth"].values[0]],
        sectoral_growth=sectoral_growth_during.values[0],
        total_firm_deposits=[total_firm_deposits_and_debt_during["Total Deposits"].values[0]],
        total_firm_debts=[total_firm_deposits_and_debt_during["Total Debt"].values[0]],
        sectoral_output=iot_industry_data_during["Output in LCU"].values[0],
        sectoral_household_consumption=iot_industry_data_during["Household Consumption in LCU"].values[0],
        sectoral_government_consumption=iot_industry_data_during["Government Consumption in LCU"].values[0],
        sectoral_imports=iot_industry_data_during["Imports in LCU"].values[0],
        sectoral_exports=iot_industry_data_during["Exports in LCU"].values[0],
        exchange_rate=[exchange_rates_data_during.values[0]],
    )
    for c in all_country_names:
        if c == country_name:
            continue
        exog_ts.dicts["sectoral_imports_from_" + c] = [iot_industry_data_during["Imports in LCU from " + c].values[0]]
        exog_ts.dicts["sectoral_exports_to_" + c] = [iot_industry_data_during["Exports in LCU to " + c].values[0]]

    # Fill
    offset = 0
    for t in range(1, len(log_inflation_during["Real CPI Inflation"].values) - offset):
        exog_ts.cpi_inflation.append([log_inflation_during["Real CPI Inflation"].values[t]])
    for t in range(1, len(log_inflation_during["Real PPI Inflation"].values) - offset):
        exog_ts.ppi_inflation.append([log_inflation_during["Real PPI Inflation"].values[t]])
    for t in range(1, len(unemployment_rate_during.values) - offset):
        exog_ts.unemployment_rate.append([unemployment_rate_during.values[t]])
    for t in range(1, len(vacancy_rate_during.values) - offset):
        exog_ts.vacancy_rate.append([vacancy_rate_during.values[t]])
    for t in range(1, len(house_price_index_during["Real House Price Index Growth"].values) - offset):
        exog_ts.real_house_price_index_growth.append(
            [house_price_index_during["Real House Price Index Growth"].values[t]]
        )
    for t in range(1, len(house_price_index_during["Nominal House Price Index Growth"].values) - offset):
        exog_ts.nominal_house_price_index_growth.append(
            [house_price_index_during["Nominal House Price Index Growth"].values[t]]
        )
    for t in range(1, len(sectoral_growth_during.values) - offset):
        exog_ts.sectoral_growth.append(sectoral_growth_during.values[t])
    for t in range(1, len(total_firm_deposits_and_debt_during["Total Deposits"].values) - offset):
        exog_ts.total_firm_deposits.append([total_firm_deposits_and_debt_during["Total Deposits"].values[t]])
    for t in range(1, len(total_firm_deposits_and_debt_during["Total Debt"].values) - offset):
        exog_ts.total_firm_debts.append([total_firm_deposits_and_debt_during["Total Debt"].values[t]])
    for t in range(1, len(iot_industry_data_during["Output in LCU"].values) - offset):
        exog_ts.sectoral_output.append(iot_industry_data_during["Output in LCU"].values[t])
    for t in range(1, len(iot_industry_data_during["Household Consumption in LCU"].values) - offset):
        exog_ts.sectoral_household_consumption.append(
            iot_industry_data_during["Household Consumption in LCU"].values[t]
        )
    for t in range(1, len(iot_industry_data_during["Government Consumption in LCU"].values) - offset):
        exog_ts.sectoral_government_consumption.append(
            iot_industry_data_during["Government Consumption in LCU"].values[t]
        )
    for t in range(1, len(iot_industry_data_during["Imports in LCU"].values) - offset):
        exog_ts.sectoral_imports.append(iot_industry_data_during["Imports in LCU"].values[t])
        for c in all_country_names:
            if c == country_name:
                continue
            exog_ts.dicts["sectoral_imports_from_" + c].append(
                iot_industry_data_during["Imports in LCU from " + c].values[t]
            )
    for t in range(1, len(iot_industry_data_during["Exports in LCU"].values) - offset):
        exog_ts.sectoral_exports.append(iot_industry_data_during["Exports in LCU"].values[t])
        for c in all_country_names:
            if c == country_name:
                continue
            exog_ts.dicts["sectoral_exports_to_" + c].append(
                iot_industry_data_during["Exports in LCU to " + c].values[t]
            )
    for t in range(1, len(exchange_rates_data_during.values)):
        num = 12 if t > 1 else 11
        for _ in range(num):
            exog_ts.exchange_rate.append([exchange_rates_data_during.values[t - 1]])

    return exog_ts
