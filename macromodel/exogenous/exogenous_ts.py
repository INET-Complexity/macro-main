import pandas as pd

from macromodel.timeseries import TimeSeries


def create_exogenous_timeseries(
    inflation_during: pd.DataFrame,
    national_accounts_during: pd.DataFrame,
    unemployment_rate_during: pd.DataFrame,
    vacancy_rate_during: pd.DataFrame,
    house_price_index_during: pd.DataFrame,
    exchange_rates_data_during: pd.DataFrame,
) -> TimeSeries:
    has_no_na = len(national_accounts_during) == 0
    exog_ts = TimeSeries(
        cpi_inflation=[inflation_during["CPI Inflation"].values[0]],
        ppi_inflation=[inflation_during["PPI Inflation"].values[0]],
        cpi=[1.0],
        ppi=[1.0],
        unemployment_rate=[unemployment_rate_during.values[0]],
        vacancy_rate=[vacancy_rate_during.values[0]],
        real_house_price_index_growth=[house_price_index_during["Real House Price Index Growth"].values[0]],
        hpi_inflation=[house_price_index_during["Nominal House Price Index Growth"].values[0]],
        gross_output=([None] if has_no_na else [national_accounts_during["Gross Output (Value)"].values[0]]),
        gross_output_growth=([None] if has_no_na else [national_accounts_during["Gross Output (Growth)"].values[0]]),
        total_household_consumption=(
            [None] if has_no_na else [national_accounts_during["Household Consumption (Value)"].values[0]]
        ),
        total_household_consumption_growth=(
            [None] if has_no_na else [national_accounts_during["Household Consumption (Growth)"].values[0]]
        ),
        total_real_government_consumption=(
            [None] if has_no_na else [national_accounts_during["Real Government Consumption (Value)"].values[0]]
        ),
        total_nominal_government_consumption=(
            [None] if has_no_na else [national_accounts_during["Government Consumption (Value)"].values[0]]
        ),
        total_nominal_government_consumption_growth=(
            [None] if has_no_na else [national_accounts_during["Government Consumption (Growth)"].values[0]]
        ),
        total_real_government_consumption_growth=(
            [None] if has_no_na else [national_accounts_during["Real Government Consumption (Growth)"].values[0]]
        ),
        total_imports=([None] if has_no_na else [national_accounts_during["Imports (Value)"].values[0]]),
        total_exports=([None] if has_no_na else [national_accounts_during["Exports (Value)"].values[0]]),
        exchange_rate=[exchange_rates_data_during.values[0]],
    )

    # Fill
    offset = 0
    for t in range(1, len(inflation_during["CPI Inflation"].values) - offset):
        exog_ts.cpi_inflation.append([inflation_during["CPI Inflation"].values[t]])
        exog_ts.cpi.append([exog_ts.current("cpi")[0] * (1 + inflation_during["CPI Inflation"].values[t])])
    for t in range(1, len(inflation_during["PPI Inflation"].values) - offset):
        exog_ts.ppi_inflation.append([inflation_during["PPI Inflation"].values[t]])
        exog_ts.ppi.append([exog_ts.current("ppi")[0] * (1 + inflation_during["PPI Inflation"].values[t])])
    for t in range(1, len(unemployment_rate_during.values) - offset):
        exog_ts.unemployment_rate.append([unemployment_rate_during.values[t]])
    for t in range(1, len(vacancy_rate_during.values) - offset):
        exog_ts.vacancy_rate.append([vacancy_rate_during.values[t]])
    for t in range(
        1,
        len(house_price_index_during["Real House Price Index Growth"].values) - offset,
    ):
        exog_ts.real_house_price_index_growth.append(
            [house_price_index_during["Real House Price Index Growth"].values[t]]
        )
    for t in range(
        1,
        len(house_price_index_during["Nominal House Price Index Growth"].values) - offset,
    ):
        exog_ts.hpi_inflation.append([house_price_index_during["Nominal House Price Index Growth"].values[t]])
    if len(national_accounts_during) > 0:
        for t in range(
            1,
            len(national_accounts_during["Gross Output (Value)"].values) - offset,
        ):
            exog_ts.gross_output.append([national_accounts_during["Gross Output (Value)"].values[t]])
            exog_ts.gross_output_growth.append([national_accounts_during["Gross Output (Growth)"].values[t]])
        for t in range(
            1,
            len(national_accounts_during["Household Consumption (Value)"].values) - offset,
        ):
            exog_ts.total_household_consumption.append(
                [national_accounts_during["Household Consumption (Value)"].values[t]]
            )
            exog_ts.total_household_consumption_growth.append(
                [national_accounts_during["Household Consumption (Growth)"].values[t]]
            )
        for t in range(
            1,
            len(national_accounts_during["Government Consumption (Value)"].values) - offset,
        ):
            exog_ts.total_real_government_consumption.append(
                [national_accounts_during["Real Government Consumption (Value)"].values[t]]
            )
            exog_ts.total_nominal_government_consumption.append(
                [national_accounts_during["Government Consumption (Value)"].values[t]]
            )
            exog_ts.total_nominal_government_consumption_growth.append(
                [national_accounts_during["Government Consumption (Growth)"].values[t]]
            )
        for t in range(1, len(national_accounts_during["Imports (Value)"].values) - offset):
            exog_ts.total_imports.append([national_accounts_during["Imports (Value)"].values[t]])
        for t in range(1, len(national_accounts_during["Exports (Value)"].values) - offset):
            exog_ts.total_exports.append([national_accounts_during["Exports (Value)"].values[t]])
    for t in range(1, len(exchange_rates_data_during.values)):
        num = 4 if t > 1 else 3
        for _ in range(num):
            exog_ts.exchange_rate.append([exchange_rates_data_during.values[t - 1]])

    return exog_ts
