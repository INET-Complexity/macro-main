import os
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from inet_data.readers.criticality_data.goods_criticality_reader import (
    GoodsCriticalityReader,
)
from inet_data.readers.economic_data.eurostat_reader import EuroStatReader
from inet_data.readers.economic_data.exchange_rates import WorldBankRatesReader
from inet_data.readers.economic_data.imf_reader import IMFReader
from inet_data.readers.economic_data.oecd_economic_data import OECDEconData
from inet_data.readers.economic_data.ons_reader import ONSReader
from inet_data.readers.economic_data.policy_rates import PolicyRatesReader
from inet_data.readers.economic_data.world_bank_reader import WorldBankReader
from inet_data.readers.io_tables.icio_reader import ICIOReader
from inet_data.readers.population_data.hfcs_reader import HFCSReader
from inet_data.readers.socioeconomic_data.wiod_sea_data import WIODSEAReader
from inet_data.readers.util.matching_iot_with_sea import (
    create_investment_matrix,
    matching_iot_with_sea,
)


class DataFilterWarning(Warning):
    pass


def get_reader_names() -> list[str]:
    return [
        "eurostat",
        "exchange_rates",
        "hfcs",
        "icio",
        "ihs_markit_goods_criticality",
        "imf",
        "inventory_to_sales_ratio",
        "notation",
        "oecd_econ",
        "ons",
        "policy_rates",
        "who",
        "wiod_sea",
        "world_bank",
    ]


def filter_columns_by_date(columns: list[str], date: str, dataset_name: Optional[str] = None) -> list[str]:
    """
    Returns
     1. All the columns that cannot be parsed as a date,
     2. The columns that can be parsed as a date and are greater than or equal to the date provided
    """
    # Identify non-date columns
    non_date_cols = [col for col in columns if pd.to_datetime(col, errors="coerce") is pd.NaT]
    # Identify date columns that are greater than or equal to x
    date_cols_to_keep = [
        col for col in columns if pd.to_datetime(col, errors="coerce") is not pd.NaT and col >= f"{date}"
    ]
    if not date_cols_to_keep:
        warnings.warn(
            f"{dataset_name}: No columns were kept for date {date}.",
            DataFilterWarning,
        )
    return non_date_cols + date_cols_to_keep


def prune_world_bank(world_bank, start_date):
    # World Bank
    for key, value in world_bank.data.items():
        years_as_columns = True
        for col in value.columns:
            if col.lower() in ["year", "time"]:
                years_as_columns = False
                # Check if column can be transformed in a date
                dates = (
                    value[col]
                    .astype(str)
                    .apply(lambda x: x.replace("Q1", "01").replace("Q2", "04").replace("Q3", "07").replace("Q4", "10"))
                )
                dates = pd.to_datetime(dates, errors="coerce")
                if dates.isnull().sum() == 0:
                    mask = dates >= pd.to_datetime(f"{start_date}-01-01")
                    if mask.sum() == 0:
                        warnings.warn(
                            f"No rows were kept for date {start_date} in World Bank dataset {key}.",
                            DataFilterWarning,
                        )
                    world_bank.data[key] = value.loc[mask, :]
                    break

        if years_as_columns is True:
            mask = filter_columns_by_date(value.columns, start_date, "World Bank")
            world_bank.data[key] = value.loc[:, mask]
    return world_bank


def prune_wiod_sea(wiod_sea, start_date):
    # WIOD_SEA
    mask = filter_columns_by_date(wiod_sea.exchange_rates.df.columns, start_date, "WIOD_SEA")
    wiod_sea.exchange_rates.df = wiod_sea.exchange_rates.df.loc[:, mask]
    return wiod_sea


def prune_imf(imf_reader, start_date):
    # IMF
    mask = filter_columns_by_date(imf_reader.data["bank_demography"].columns, start_date, "IMF")
    imf_reader.data["bank_demography"] = imf_reader.data["bank_demography"].loc[:, mask]
    return imf_reader


def prune_policy_rates(policy_rates, start_date):
    # Policy rates
    mask = filter_columns_by_date(policy_rates.df.columns, start_date, "Policy rates")
    policy_rates.df = policy_rates.df.loc[:, mask]
    return policy_rates


def prune_oecd(oecd_econ, start_date):
    # OECD
    for key, value in oecd_econ.data.items():
        for col in value.columns:
            if col.lower() in ["year", "time"]:
                # Check if column can be transformed in a date
                dates = pd.to_datetime(value[col].astype(str), errors="coerce")
                if dates.isnull().sum() == 0:
                    mask = dates >= pd.to_datetime(f"{start_date}-01-01")
                    if mask.sum() == 0:
                        warnings.warn(
                            f"No rows after {start_date} in OECD dataset {key}; No filter applied.",
                            DataFilterWarning,
                        )
                        mask = np.ones(len(value), dtype=bool)
                    oecd_econ.data[key] = value.loc[mask, :]
                    break
            if col == "country_year":
                mask = value[col].apply(lambda x: x.split("_")[1]) >= f"{start_date}"
                if mask.sum() == 0:
                    warnings.warn(
                        f"No rows were kept for date {start_date} in OECD dataset {key}.",
                        DataFilterWarning,
                    )
                oecd_econ.data[key] = value.loc[mask, :]
                break
    return oecd_econ


def prune_icio(icio, start_date):
    # ICIO
    for key, value in icio.items():
        if f"{key}".isnumeric() and f"{key}" < f"{start_date}":
            _ = icio.pop(key, None)
    if not icio:
        warnings.warn(
            f"No ICIO data was kept for date {start_date}.",
            DataFilterWarning,
        )
    return icio


def prune_eurostat(eurostat, start_date):
    # Eurostat
    for key, value in eurostat.data.items():
        if "TIME_PERIOD" in value.columns:
            mask = value["TIME_PERIOD"].astype(str) >= str(start_date)
            if mask.sum() == 0:
                warnings.warn(
                    f"No rows were kept for date {start_date} in Eurostat dataset {key}.",
                    DataFilterWarning,
                )
            eurostat.data[key] = value.loc[mask, :]
        else:
            mask = filter_columns_by_date(value.columns, start_date, f"Eurostat {key}")
            eurostat.data[key] = value.loc[:, mask]
    return eurostat


def prune_wb_exchange_rates(exchange_rates, start_date):
    # WB exchange rates
    mask = filter_columns_by_date(exchange_rates.df.columns, start_date, "WB exchange rates")
    exchange_rates.df = exchange_rates.df.loc[:, mask]
    return exchange_rates


def prune_data(
    start_date: str,
    exchange_rates: WorldBankRatesReader,
    eurostat: EuroStatReader,
    icio: dict[str, Any],
    oecd_econ: OECDEconData,
    policy_rates: PolicyRatesReader,
    imf_reader: IMFReader,
    wiod_sea: WIODSEAReader,
    world_bank: WorldBankReader,
):
    """
    Removes data before start_date
    """
    exchange_rates = prune_wb_exchange_rates(exchange_rates, start_date)
    eurostat = prune_eurostat(eurostat, start_date)
    icio = prune_icio(icio, start_date)
    oecd_econ = prune_oecd(oecd_econ, start_date)
    policy_rates = prune_policy_rates(policy_rates, start_date)
    imf_reader = prune_imf(imf_reader, start_date)
    wiod_sea = prune_wiod_sea(wiod_sea, start_date)
    world_bank = prune_world_bank(world_bank, start_date)

    return exchange_rates, eurostat, icio, oecd_econ, policy_rates, imf_reader, wiod_sea, world_bank


def init_readers(
    raw_data_path: Path,
    country_names: list[str],
    country_names_short: list[str],
    year: int,
    scale: int,
    industries: list[str],
    start_date: Optional[str] = None,
    create_exogenous_industry_data: bool = False,
    testing: bool = False,
) -> dict[str, Any]:
    # Goods criticality reader
    goods_criticality = GoodsCriticalityReader.from_csv(
        path=raw_data_path / "ihs_markit_goods_criticality" / "UK_2020.csv",
    )

    # Exchange rates
    exchange_rates = WorldBankRatesReader.from_csv(path=raw_data_path / "exchange_rates" / "exchange_rates.csv")

    # Eurostat inet_data
    eurostat = EuroStatReader(
        path=raw_data_path / "eurostat",
        country_code_path=raw_data_path / "notation" / "wikipedia-iso-country-codes.csv",
    )

    # Population microdata
    hfcs = {}
    for ind, country_name in enumerate(country_names):
        hfcs[country_name] = HFCSReader.from_csv(
            country_name=country_name,
            country_name_short=country_names_short[ind],
            year=year,
            hfcs_data_path=raw_data_path / "hfcs",
            exchange_rates=exchange_rates,
            num_surveys=1 if testing else 5,
        )

    # Input-Output Tables
    icio = {}
    if create_exogenous_industry_data:
        icio_years = range(2010, 2019)
    else:
        icio_years = [year]
    for iot_year in tqdm(icio_years, desc="Reading ICIO Tables"):
        path = raw_data_path / "icio" / str(iot_year) / ("ICIO2021_" + str(iot_year) + ".csv")
        if os.path.isfile(path):
            icio[iot_year] = ICIOReader.agg_from_csv(
                path=path,
                pivot_path=raw_data_path / "icio" / str(iot_year) / ("ICIO2021_" + str(iot_year) + "_P.csv"),
                considered_countries=country_names,
                aggregation_path=raw_data_path / "icio" / "mappings.json",
                industries=industries,
                year=iot_year,
                exchange_rates=exchange_rates,
                imputed_rent_fraction=eurostat.get_imputed_rent_fraction(country_names, 2014),
            )

    # Socio-Economic Accounts
    wiod_sea = WIODSEAReader.agg_from_csv(
        path=raw_data_path / "wiod_sea" / "wiod_sea.csv",
        aggregation_path=raw_data_path / "wiod_sea" / "mappings.json",
        year=year,
        country_names=country_names,
        industries=industries,
        exchange_rates=exchange_rates,
    )

    # Create a proportional investment matrix
    create_investment_matrix(
        icio_reader=icio[year],
        sea_reader=wiod_sea,
        country_names=country_names,
    )

    # Match IOT with SEA inet_data
    matching_iot_with_sea(
        icio_reader=icio[year],
        sea_reader=wiod_sea,
        country_names=country_names,
    )

    # OECD inet_data
    oecd_econ = OECDEconData(
        path=raw_data_path / "oecd_econ",
        industry_mappings_path=raw_data_path / "oecd_econ" / "mappings.json",
        sector_mapping_path=raw_data_path / "icio" / "mappings.json",
        scale=scale,
    )

    # Policy rates
    policy_rates = PolicyRatesReader(
        path=raw_data_path / "policy_rates" / "bis_cb_policy_rates.csv",
        country_code_path=raw_data_path / "notation" / "wikipedia-iso-country-codes.csv",
    )

    # IMF inet_data
    imf_reader = IMFReader(path=raw_data_path / "imf", scale=scale)

    # World Bank inet_data
    world_bank = WorldBankReader(path=raw_data_path / "world_bank")

    # ONS Firm Data
    ons_reader = ONSReader(path=raw_data_path / "ons")

    if start_date is not None:
        exchange_rates, eurostat, icio, oecd_econ, policy_rates, imf_reader, wiod_sea, world_bank = prune_data(
            exchange_rates=exchange_rates,
            eurostat=eurostat,
            icio=icio,
            oecd_econ=oecd_econ,
            policy_rates=policy_rates,
            imf_reader=imf_reader,
            wiod_sea=wiod_sea,
            world_bank=world_bank,
            start_date=start_date,
        )

    return {
        "goods_criticality": goods_criticality,
        "exchange_rates": exchange_rates,
        "eurostat": eurostat,
        "hfcs": hfcs,
        "icio": icio,
        "oecd_econ": oecd_econ,
        "policy_rates": policy_rates,
        "imf_reader": imf_reader,
        "wiod_sea": wiod_sea,
        "world_bank": world_bank,
        "ons": ons_reader,
    }
