import numpy as np
import pandas as pd

from inet_data.readers.default_readers import DataReaders
from inet_data.readers.economic_data.exchange_rates import WorldBankRatesReader
from inet_data.readers.economic_data.oecd_economic_data import OECDEconData
from inet_data.readers.io_tables.icio_reader import ICIOReader
from inet_data.readers.socioeconomic_data.wiod_sea_data import WIODSEAReader


def compile_industry_data(
    year: int,
    readers: DataReaders,
    country_names: list[str],
    single_firm_per_industry: dict[str, bool],
) -> dict[str, dict[str, pd.DataFrame]]:
    industry_data = {}
    current_icio_reader = readers.icio[year]
    sea_reader = readers.wiod_sea
    exchange_rates = readers.exchange_rates
    econ_reader = readers.oecd_econ

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
        industry_vectors = get_industry_vectors(
            country_name,
            current_icio_reader,
            exchange_rate,
            sea_reader,
            econ_reader,
            single_firm_per_industry[country_name],
            trade_partners=country_names + ["ROW"],
        )

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

    fill_trade_data(
        "ROW", current_icio_reader, exchange_rate_row, industry_data["ROW"]["industry_vectors"], country_names
    )

    return industry_data


def get_industry_vectors(
    country_name: str,
    current_icio_reader: ICIOReader,
    exchange_rate: float,
    sea_reader: WIODSEAReader,
    econ_reader: OECDEconData,
    single_firm_per_industry: bool,
    trade_partners: list[str],
) -> pd.DataFrame:
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
    # Record the number of firms by sector
    if single_firm_per_industry:
        industry_vectors["Number of Firms"] = np.ones(len(current_icio_reader.industries), dtype=int)
    else:
        n_firms_by_industry = econ_reader.read_business_demography(
            country=country_name,
            output=pd.Series(industry_vectors["Output"].values),
            year=sea_reader.year,
        )
        n_firms_by_industry[n_firms_by_industry == 0] = 1
        industry_vectors["Number of Firms"] = n_firms_by_industry

    fill_trade_data(country_name, current_icio_reader, exchange_rate, industry_vectors, trade_partners)

    return industry_vectors


def fill_trade_data(
    country_name: str,
    current_icio_reader: ICIOReader,
    exchange_rate: float,
    industry_vectors: pd.DataFrame,
    trade_partners: list[str],
):
    for c in trade_partners:
        if c == country_name:
            continue
        industry_vectors["Exports in USD to " + c] = current_icio_reader.get_trade(country_name, c)
        industry_vectors["Exports in LCU to " + c] = exchange_rate * current_icio_reader.get_trade(country_name, c)
        industry_vectors["Imports in USD from " + c] = current_icio_reader.get_trade(c, country_name)
        industry_vectors["Imports in LCU from " + c] = exchange_rate * current_icio_reader.get_trade(c, country_name)


def compile_exogenous_industry_data(
    readers: DataReaders, country_names: list[str], year_min: int = 2010, year_max: int = 2019
) -> dict[str, pd.DataFrame]:
    icio_readers = readers.icio
    exchange_rates = readers.exchange_rates

    # Handle regular countries
    exogenous_industry_data = {
        country: pd.concat(
            [
                get_country_industry_data(country, country_names, exchange_rates, icio_readers, year)
                for year in range(year_min, year_max)
                if year in icio_readers.keys()
            ]
        )
        for country in country_names
    }

    # Adding the rest of the world
    exogenous_industry_data["ROW"] = pd.concat(
        [
            get_row_industry_data(country_names, exchange_rates, icio_readers, year)
            for year in range(year_min, year_max)
            if year in icio_readers.keys()
        ]
    )

    # Interpolate if we have enough data
    for country_name, country_data in exogenous_industry_data.items():
        if country_data.shape[0] > 1:
            country_data = country_data.astype(float).resample("M").interpolate("linear").copy()
            country_data.index = pd.DatetimeIndex([pd.Timestamp(d.year, d.month, 1) for d in country_data.index])
            exogenous_industry_data[country_name] = country_data

    return exogenous_industry_data


def get_row_industry_data(
    country_names: list[str], exchange_rates: WorldBankRatesReader, icio_readers: dict[int, ICIOReader], year: int
) -> pd.DataFrame:
    exchange_rate_row = exchange_rates.from_usd_to_lcu("ROW", year)
    row_industry_data = pd.DataFrame(
        data={
            "Exports in USD": icio_readers[year].get_exports("ROW"),
            "Exports in LCU": exchange_rate_row * icio_readers[year].get_exports("ROW"),
            "Imports in USD": icio_readers[year].get_imports("ROW"),
            "Imports in LCU": exchange_rate_row * icio_readers[year].get_imports("ROW"),
        },
        index=pd.Index(icio_readers[year].industries, name="Industry"),
    )
    for c in country_names:
        row_industry_data["Exports in USD to " + c] = icio_readers[year].get_trade("ROW", c)
        row_industry_data["Exports in LCU to " + c] = exchange_rate_row * icio_readers[year].get_trade("ROW", c)
        row_industry_data["Imports in USD from " + c] = icio_readers[year].get_trade(c, "ROW")
        row_industry_data["Imports in LCU from " + c] = exchange_rate_row * icio_readers[year].get_trade(c, "ROW")

    row_industry_data = row_industry_data.stack().to_frame().T.swaplevel(0, 1, axis=1)

    row_industry_data.index = [pd.to_datetime(year, format="%Y")]

    return row_industry_data


def get_country_industry_data(
    country_name: str,
    country_names: list[str],
    exchange_rates: WorldBankRatesReader,
    icio_readers: dict[int, ICIOReader],
    year: int,
) -> pd.DataFrame:
    # Exchange rates
    exchange_rate = exchange_rates.from_usd_to_lcu(country_name, year)
    # Industry vectors
    industry_data = pd.DataFrame(
        data={
            "Output in USD": icio_readers[year].get_monthly_total_output(country_name),
            "Output in LCU": exchange_rate * icio_readers[year].get_monthly_total_output(country_name),
            "Intermediate Inputs Supply": icio_readers[year]
            .get_monthly_intermediate_inputs_use(country_name)
            .sum(axis=0),
            "Intermediate Inputs Use": icio_readers[year].get_monthly_intermediate_inputs_use(country_name).sum(axis=1),
            "Intermediate Inputs Domestic Use": icio_readers[year]
            .get_monthly_intermediate_inputs_domestic(country_name)
            .sum(axis=1),
            "Capital Inputs Domestic": icio_readers[year].get_monthly_capital_inputs_domestic(country_name),
            "Capital Inputs": icio_readers[year].get_monthly_capital_inputs(country_name),
            "Value Added in USD": icio_readers[year].get_monthly_value_added(country_name),
            "Value Added in LCU": exchange_rate * icio_readers[year].get_monthly_value_added(country_name),
            "Taxes Less Subsidies Rates": icio_readers[year].get_taxes_less_subsidies_rates(country_name),
            "Household Consumption in USD": icio_readers[year].get_monthly_hh_consumption(country_name),
            "Household Consumption in LCU": exchange_rate * icio_readers[year].get_monthly_hh_consumption(country_name),
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
        industry_data["Exports in USD to " + c] = icio_readers[year].get_trade(country_name, c)
        industry_data["Exports in LCU to " + c] = exchange_rate * icio_readers[year].get_trade(country_name, c)
        industry_data["Imports in USD from " + c] = icio_readers[year].get_trade(c, country_name)
        industry_data["Imports in LCU from " + c] = exchange_rate * icio_readers[year].get_trade(c, country_name)

    # put everything in one row
    industry_data = industry_data.stack().to_frame().T.swaplevel(0, 1, axis=1)

    industry_data.index = [pd.to_datetime(year, format="%Y")]

    return industry_data
