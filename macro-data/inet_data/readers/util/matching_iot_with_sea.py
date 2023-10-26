from typing import Any

import numpy as np
import pandas as pd

from inet_data.readers.economic_data.exchange_rates import WorldBankRatesReader
from inet_data.readers.economic_data.oecd_economic_data import OECDEconData
from inet_data.readers.io_tables.icio_reader import ICIOReader
from inet_data.readers.socioeconomic_data.wiod_sea_data import WIODSEAReader


def get_sea(
    country_name: str,
    field: str,
    sea_reader: WIODSEAReader,
) -> np.ndarray:
    return sea_reader.df.loc[
        sea_reader.df.index.get_level_values(0) == country_name,
        field,
    ].values


def create_investment_matrix(
    icio_reader: ICIOReader,
    sea_reader: WIODSEAReader,
    country_names: list[str],
) -> None:
    for country_name in country_names:
        gfcf = icio_reader.get_monthly_capital_inputs(country_name)
        cap = sea_reader.get_values_in_usd(country_name, "Capital Compensation") / 12.0
        investment_matrix = np.array([gfcf for _ in range(len(cap))]).T
        investment_matrix = investment_matrix * cap[None, :]  # proportionally fitting CAP
        investment_matrix *= gfcf.sum() / investment_matrix.sum()  # match GFCF exactly
        investment_matrix = (
            pd.DataFrame(
                data=investment_matrix,
                index=pd.MultiIndex.from_product(
                    [[country_name], icio_reader.industries],
                    names=["Country", "Industry"],
                ),
                columns=pd.MultiIndex.from_product(
                    [[country_name], icio_reader.industries],
                    names=["Country", "Industry"],
                ),
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        icio_reader.investment_matrices[country_name] = investment_matrix


def matching_iot_with_sea(
    icio_reader: ICIOReader,
    sea_reader: WIODSEAReader,
    country_names: list[str],
) -> None:
    for country_name in country_names:
        sea_reader.df.loc[
            sea_reader.df.index.get_level_values(0) == country_name,
            "Capital Compensation",
        ] = 12 * icio_reader.investment_matrices[country_name].values.sum(axis=0)
        new_va = 12 * icio_reader.get_monthly_value_added(country_name)
        va_factor = new_va / get_sea(country_name, "Value Added", sea_reader)
        sea_reader.df.loc[
            sea_reader.df.index.get_level_values(0) == country_name,
            "Value Added",
        ] = new_va
        sea_reader.df.loc[
            sea_reader.df.index.get_level_values(0) == country_name,
            "Labour Compensation",
        ] = get_sea(
            country_name, "Value Added", sea_reader
        ) - get_sea(country_name, "Capital Compensation", sea_reader)
        sea_reader.df.loc[
            sea_reader.df.index.get_level_values(0) == country_name,
            "Capital Stock",
        ] *= va_factor


def compile_industry_data(
    current_icio_reader: ICIOReader,
    sea_reader: WIODSEAReader,
    econ_reader: OECDEconData,
    exchange_rates: WorldBankRatesReader,
    country_names: list[str],
    config: dict[str, Any],
) -> dict[str, dict[str, pd.DataFrame]]:
    industry_data = {}
    for country_name in country_names:
        # Matrices
        intermediate_inputs_productivity_matrix = current_icio_reader.get_intermediate_inputs_matrix(country_name)
        capital_inputs_productivity_matrix = current_icio_reader.get_capital_inputs_matrix(
            country_name=country_name,
            capital_stock=sea_reader.get_values_in_usd(country_name, "Capital Stock"),
        )
        capital_inputs_depreciation_matrix = current_icio_reader.get_capital_inputs_depreciation(
            country_name=country_name,
            capital_compensation=sea_reader.get_values_in_usd(country_name, "Capital Compensation"),
        )

        # Exchange rates
        exchange_rate = exchange_rates.from_usd_to_lcu(country_name, sea_reader.year)

        # Industry vectors
        industry_vectors = pd.DataFrame(
            data={
                "Output": current_icio_reader.get_monthly_total_output(country_name),
                "Intermediate Inputs Supply": current_icio_reader.get_monthly_intermediate_inputs_use(country_name).sum(
                    axis=0
                ),
                "Intermediate Inputs Use": current_icio_reader.get_monthly_intermediate_inputs_use(country_name).sum(
                    axis=1
                ),
                "Intermediate Inputs Domestic Use": current_icio_reader.get_monthly_intermediate_inputs_domestic(
                    country_name
                ).sum(axis=1),
                "Capital Inputs Domestic": current_icio_reader.get_monthly_capital_inputs_domestic(country_name),
                "Capital Inputs": current_icio_reader.get_monthly_capital_inputs(country_name),
                "Value Added in USD": current_icio_reader.get_monthly_value_added(country_name),
                "Value Added in LCU": exchange_rate * current_icio_reader.get_monthly_value_added(country_name),
                "Taxes Less Subsidies in USD": current_icio_reader.get_monthly_taxes_less_subsidies(country_name),
                "Taxes Less Subsidies Rates": current_icio_reader.get_taxes_less_subsidies_rates(country_name),
                "Household Consumption in USD": current_icio_reader.get_monthly_hh_consumption(country_name),
                "Household Consumption in LCU": exchange_rate
                * current_icio_reader.get_monthly_hh_consumption(country_name),
                "Domestic Household Consumption in USD": current_icio_reader.get_monthly_hh_consumption_domestic(
                    country_name
                ),
                "Domestic Household Consumption in LCU": exchange_rate
                * current_icio_reader.get_monthly_hh_consumption_domestic(country_name),
                "Household Consumption Weights": current_icio_reader.get_hh_consumption_weights(country_name),
                "Government Consumption in USD": current_icio_reader.get_monthly_govt_consumption(country_name),
                "Government Consumption in LCU": exchange_rate
                * current_icio_reader.get_monthly_govt_consumption(country_name),
                "Domestic Government Consumption in USD": current_icio_reader.get_monthly_govt_consumption_domestic(
                    country_name
                ),
                "Domestic Government Consumption in LCU": exchange_rate
                * current_icio_reader.get_monthly_govt_consumption_domestic(country_name),
                "Government Consumption Weights": current_icio_reader.govt_consumption_weights(country_name),
                "Labour Compensation in USD": sea_reader.get_values_in_usd(country_name, "Labour Compensation") / 12.0,
                "Labour Compensation in LCU": sea_reader.get_values_in_lcu(country_name, "Labour Compensation") / 12.0,
                "Capital Stock": sea_reader.get_values_in_usd(country_name, "Capital Stock"),
                "Average Initial Price": np.full(len(current_icio_reader.industries), exchange_rate),
                "Exports in USD": current_icio_reader.get_exports(country_name),
                "Exports in LCU": exchange_rate * current_icio_reader.get_exports(country_name),
                "Imports in USD": current_icio_reader.get_imports(country_name),
                "Imports in LCU": exchange_rate * current_icio_reader.get_imports(country_name),
            },
            index=pd.Index(current_icio_reader.industries, name="Industry"),
        )
        for c in country_names + ["ROW"]:
            if c == country_name:
                continue
            industry_vectors["Exports in USD to " + c] = current_icio_reader.get_trade(country_name, c)
            industry_vectors["Exports in LCU to " + c] = exchange_rate * current_icio_reader.get_trade(country_name, c)
            industry_vectors["Imports in USD from " + c] = current_icio_reader.get_trade(c, country_name)
            industry_vectors["Imports in LCU from " + c] = exchange_rate * current_icio_reader.get_trade(
                c, country_name
            )

        # Record the number of firms by sector
        if config["model"]["single_firm_per_industry"]["value"]:
            industry_vectors["Number of Firms"] = np.ones(len(current_icio_reader.industries), dtype=int)
        else:
            n_firms_by_industry = econ_reader.read_business_demography(
                country=country_name,
                output=pd.Series(industry_vectors["Output"].values),
                year=sea_reader.year,
            )
            n_firms_by_industry[n_firms_by_industry == 0] = 1
            industry_vectors["Number of Firms"] = n_firms_by_industry

        # Record all
        industry_data[country_name] = {
            "intermediate_inputs_productivity_matrix": intermediate_inputs_productivity_matrix,
            "capital_inputs_productivity_matrix": capital_inputs_productivity_matrix,
            "capital_inputs_depreciation_matrix": capital_inputs_depreciation_matrix,
            "industry_vectors": industry_vectors,
        }

    # Adding the rest of the world
    exchange_rate_row = exchange_rates.from_usd_to_lcu("ROW", sea_reader.year)
    industry_data["ROW"] = {
        "industry_vectors": {
            "Exports in USD": current_icio_reader.get_exports("ROW"),
            "Exports in LCU": exchange_rate_row * current_icio_reader.get_exports("ROW"),
            "Imports in USD": current_icio_reader.get_imports("ROW"),
            "Imports in LCU": exchange_rate_row * current_icio_reader.get_imports("ROW"),
        }
    }
    for c in country_names:
        industry_data["ROW"]["industry_vectors"]["Exports in USD to " + c] = current_icio_reader.get_trade("ROW", c)
        industry_data["ROW"]["industry_vectors"][
            "Exports in LCU to " + c
        ] = exchange_rate_row * current_icio_reader.get_trade("ROW", c)
        industry_data["ROW"]["industry_vectors"]["Imports in USD from " + c] = current_icio_reader.get_trade(c, "ROW")
        industry_data["ROW"]["industry_vectors"][
            "Imports in LCU from " + c
        ] = exchange_rate_row * current_icio_reader.get_trade(c, "ROW")

    return industry_data


def compile_exogenous_industry_data(
    icio_readers: dict[int, ICIOReader],
    exchange_rates: WorldBankRatesReader,
    country_names: list[str],
) -> dict[str, pd.DataFrame]:
    exogenous_industry_data = {}

    # Handle regular countries
    for country_name in country_names:
        exogenous_industry_data[country_name] = {}
        for year in range(2010, 2019):
            if year not in icio_readers.keys():
                continue

            # Exchange rates
            exchange_rate = exchange_rates.from_usd_to_lcu(country_name, year)

            # Industry vectors
            exogenous_industry_data[country_name][year] = pd.DataFrame(
                data={
                    "Output in USD": icio_readers[year].get_monthly_total_output(country_name),
                    "Output in LCU": exchange_rate * icio_readers[year].get_monthly_total_output(country_name),
                    "Intermediate Inputs Supply": icio_readers[year]
                    .get_monthly_intermediate_inputs_use(country_name)
                    .sum(axis=0),
                    "Intermediate Inputs Use": icio_readers[year]
                    .get_monthly_intermediate_inputs_use(country_name)
                    .sum(axis=1),
                    "Intermediate Inputs Domestic Use": icio_readers[year]
                    .get_monthly_intermediate_inputs_domestic(country_name)
                    .sum(axis=1),
                    "Capital Inputs Domestic": icio_readers[year].get_monthly_capital_inputs_domestic(country_name),
                    "Capital Inputs": icio_readers[year].get_monthly_capital_inputs(country_name),
                    "Value Added in USD": icio_readers[year].get_monthly_value_added(country_name),
                    "Value Added in LCU": exchange_rate * icio_readers[year].get_monthly_value_added(country_name),
                    "Taxes Less Subsidies Rates": icio_readers[year].get_taxes_less_subsidies_rates(country_name),
                    "Household Consumption in USD": icio_readers[year].get_monthly_hh_consumption(country_name),
                    "Household Consumption in LCU": exchange_rate
                    * icio_readers[year].get_monthly_hh_consumption(country_name),
                    "Domestic Household Consumption in USD": icio_readers[year].get_monthly_hh_consumption_domestic(
                        country_name
                    ),
                    "Domestic Household Consumption in LCU": exchange_rate
                    * icio_readers[year].get_monthly_hh_consumption_domestic(country_name),
                    "Household Consumption Weights": icio_readers[year].get_hh_consumption_weights(country_name),
                    "Government Consumption in USD": icio_readers[year].get_monthly_govt_consumption(country_name),
                    "Government Consumption in LCU": exchange_rate
                    * icio_readers[year].get_monthly_govt_consumption(country_name),
                    "Domestic Government Consumption in USD": icio_readers[year].get_monthly_govt_consumption_domestic(
                        country_name
                    ),
                    "Domestic Government Consumption in LCU": exchange_rate
                    * icio_readers[year].get_monthly_govt_consumption_domestic(country_name),
                    "Government Consumption Weights": icio_readers[year].govt_consumption_weights(country_name),
                    "Exports in USD": icio_readers[year].get_exports(country_name),
                    "Exports in LCU": exchange_rate * icio_readers[year].get_exports(country_name),
                    "Imports in USD": icio_readers[year].get_imports(country_name),
                    "Imports in LCU": exchange_rate * icio_readers[year].get_imports(country_name),
                    "Average Initial Price": np.full(len(icio_readers[year].industries), exchange_rate),
                },
                index=pd.Index(icio_readers[year].industries, name="Industry"),
            )
            for c in country_names + ["ROW"]:
                if c == country_name:
                    continue
                exogenous_industry_data[country_name][year]["Exports in USD to " + c] = icio_readers[year].get_trade(
                    country_name, c
                )
                exogenous_industry_data[country_name][year]["Exports in LCU to " + c] = exchange_rate * icio_readers[
                    year
                ].get_trade(country_name, c)
                exogenous_industry_data[country_name][year]["Imports in USD from " + c] = icio_readers[year].get_trade(
                    c, country_name
                )
                exogenous_industry_data[country_name][year]["Imports in LCU from " + c] = exchange_rate * icio_readers[
                    year
                ].get_trade(c, country_name)

    # Adding the rest of the world
    exogenous_industry_data["ROW"] = {}
    for year in range(2010, 2019):
        if year not in icio_readers.keys():
            continue
        exchange_rate_row = exchange_rates.from_usd_to_lcu("ROW", year)
        exogenous_industry_data["ROW"][year] = pd.DataFrame(
            data={
                "Exports in USD": icio_readers[year].get_exports("ROW"),
                "Exports in LCU": exchange_rate_row * icio_readers[year].get_exports("ROW"),
                "Imports in USD": icio_readers[year].get_imports("ROW"),
                "Imports in LCU": exchange_rate_row * icio_readers[year].get_imports("ROW"),
            },
            index=pd.Index(icio_readers[year].industries, name="Industry"),
        )
        for c in country_names:
            exogenous_industry_data["ROW"][year]["Exports in USD to " + c] = icio_readers[year].get_trade("ROW", c)
            exogenous_industry_data["ROW"][year]["Exports in LCU to " + c] = exchange_rate_row * icio_readers[
                year
            ].get_trade("ROW", c)
            exogenous_industry_data["ROW"][year]["Imports in USD from " + c] = icio_readers[year].get_trade(c, "ROW")
            exogenous_industry_data["ROW"][year]["Imports in LCU from " + c] = exchange_rate_row * icio_readers[
                year
            ].get_trade(c, "ROW")

    # Interpolate
    exogenous_data_by_country = {}
    for country_name in exogenous_industry_data.keys():
        country_data_ls = []
        for year in range(2010, 2019):
            if year not in exogenous_industry_data[country_name].keys():
                continue
            curr_data = exogenous_industry_data[country_name][year]  # noqa
            curr_data = curr_data.stack().to_frame().T
            curr_data = curr_data.swaplevel(0, 1, axis=1)
            curr_data.index = [year]
            country_data_ls.append(curr_data)
        country_data = pd.concat(country_data_ls, axis=0)
        country_data.index = pd.DatetimeIndex(pd.date_range(start="2010-01-01", end="2019-01-01", freq="Y"))
        # country_data = country_data.reindex(pd.date_range(start="2010-01-01", end="2019-01-01", freq="M"))
        country_data = country_data.astype(float).resample("M").interpolate("linear").copy()
        country_data.index = pd.DatetimeIndex([pd.Timestamp(d.year, d.month, 1) for d in country_data.index])
        exogenous_data_by_country[country_name] = country_data

    return exogenous_data_by_country
