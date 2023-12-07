import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple, Any

import pandas as pd

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
    add_investment_matrix_to_icio,
    match_iot_with_sea,
)
from dataclasses import dataclass

from inet_data.readers.util.prune_util import DataFilterWarning


@dataclass
class DataPaths:
    goods_criticality_path: Path
    exchange_rates_path: Path
    eurostat_path: Path
    hfcs_path: Path
    icio_paths: dict[int, Path]
    icio_pivot_paths: dict[int, Path]
    icio_agg_path: Path
    wiod_sea_path: Path
    wiod_sea_agg_path: Path
    oecd_econ_path: Path
    oecd_econ_mapping_path: Path
    policy_rates_path: Path
    country_codes_path: Path
    imf_path: Path
    ons_path: Path
    world_bank_path: Path

    @classmethod
    def default_paths(cls, raw_data_path: Path, icio_years: Iterable[int]):
        return cls(
            goods_criticality_path=raw_data_path / "ihs_markit_goods_criticality" / "UK_2020.csv",
            exchange_rates_path=raw_data_path / "exchange_rates" / "exchange_rates.csv",
            eurostat_path=raw_data_path / "eurostat",
            hfcs_path=raw_data_path / "hfcs",
            icio_paths={
                year: raw_data_path / "icio" / str(year) / ("ICIO2021_" + str(year) + ".csv") for year in icio_years
            },
            icio_pivot_paths={
                year: raw_data_path / "icio" / str(year) / ("ICIO2021_" + str(year) + "_pivot.csv")
                for year in icio_years
            },
            icio_agg_path=raw_data_path / "icio" / "mappings.json",
            wiod_sea_path=raw_data_path / "wiod_sea" / "wiod_sea.csv",
            wiod_sea_agg_path=raw_data_path / "wiod_sea" / "mappings.json",
            oecd_econ_path=raw_data_path / "oecd_econ",
            oecd_econ_mapping_path=raw_data_path / "oecd_econ" / "mappings.json",
            policy_rates_path=raw_data_path / "policy_rates" / "bis_cb_policy_rates.csv",
            country_codes_path=raw_data_path / "notation" / "wikipedia-iso-country-codes.csv",
            imf_path=raw_data_path / "imf",
            ons_path=raw_data_path / "ons",
            world_bank_path=raw_data_path / "world_bank",
        )


@dataclass
class DataReaders:
    icio: dict[int, ICIOReader]
    wiod_sea: WIODSEAReader
    oecd_econ: OECDEconData
    world_bank: WorldBankReader
    hfcs: dict[str, HFCSReader]
    eurostat: EuroStatReader
    ons: ONSReader
    policy_rates: PolicyRatesReader
    imf_reader: IMFReader
    exchange_rates: WorldBankRatesReader
    goods_criticality: GoodsCriticalityReader

    @classmethod
    def init_default_raw_data_path(
        cls,
        raw_data_path: Path,
        country_names: list[str],
        country_names_short: list[str],
        simulation_year: int,
        scale: int,
        industries: list[str],
        create_exogenous_industry_data: bool = False,
        imputed_rent_year: int = 2014,
        exog_data_range: Tuple[int, int] = (2010, 2018),
        prune_date: str | int | datetime = None,
        force_single_hfcs_survey: bool = False,
        prune_date_format: str = "%Y-%m-%d",
    ):
        short_names = {
            country_name: country_name_short
            for country_name, country_name_short in zip(country_names, country_names_short)
        }
        if not create_exogenous_industry_data:
            all_years = [simulation_year]
        else:
            all_years = range(exog_data_range[0], exog_data_range[1] + 1)
        datapaths = DataPaths.default_paths(raw_data_path, all_years)

        goods_criticality = GoodsCriticalityReader.from_csv(path=datapaths.goods_criticality_path)
        exchange_rates = WorldBankRatesReader.from_csv(path=datapaths.exchange_rates_path)
        eurostat = EuroStatReader(path=datapaths.eurostat_path, country_code_path=datapaths.country_codes_path)
        hfcs = {
            country_name: HFCSReader.from_csv(
                country_name=country_name,
                country_name_short=short_names[country_name],
                hfcs_data_path=datapaths.hfcs_path,
                year=simulation_year,
                exchange_rates=exchange_rates,
                num_surveys=1 if force_single_hfcs_survey else 5,
            )
            for country_name in country_names
        }

        icio = {
            year: ICIOReader.agg_from_csv(
                path=datapaths.icio_paths[year],
                pivot_path=datapaths.icio_pivot_paths[year],
                year=year,
                aggregation_path=datapaths.icio_agg_path,
                considered_countries=country_names,
                industries=industries,
                exchange_rates=exchange_rates,
                imputed_rent_fraction=eurostat.get_imputed_rent_fraction(country_names, imputed_rent_year),
            )
            for year in all_years
        }

        wiod_sea = WIODSEAReader.agg_from_csv(
            path=datapaths.wiod_sea_path,
            year=simulation_year,
            industries=industries,
            exchange_rates=exchange_rates,
            aggregation_path=datapaths.wiod_sea_agg_path,
            country_names=country_names,
        )

        add_investment_matrix_to_icio(
            icio_reader=icio[simulation_year], sea_reader=wiod_sea, country_names=country_names
        )

        match_iot_with_sea(icio_reader=icio[simulation_year], sea_reader=wiod_sea, country_names=country_names)

        oecd_econ = OECDEconData(
            path=datapaths.oecd_econ_path,
            industry_mappings_path=datapaths.oecd_econ_mapping_path,
            sector_mapping_path=datapaths.icio_agg_path,
            scale=scale,
        )

        policy_rates = PolicyRatesReader(
            path=datapaths.policy_rates_path, country_code_path=datapaths.country_codes_path
        )

        imf_reader = IMFReader(path=datapaths.imf_path, scale=scale)

        ons_reader = ONSReader(path=datapaths.ons_path)

        world_bank = WorldBankReader(path=datapaths.world_bank_path)

        if prune_date:
            exchange_rates.prune(prune_date, prune_date_format=prune_date_format)
            eurostat.prune(prune_date, prune_date_format=prune_date_format)
            icio = prune_icio_dict(icio, prune_date)
            wiod_sea.prune(prune_date, prune_date_format=prune_date_format)
            oecd_econ.prune(prune_date, prune_date_format=prune_date_format)
            policy_rates.prune(prune_date, prune_date_format=prune_date_format)
            imf_reader.prune(prune_date, prune_date_format=prune_date_format)
            world_bank.prune(prune_date, prune_date_format=prune_date_format)

        return cls(
            icio=icio,
            wiod_sea=wiod_sea,
            oecd_econ=oecd_econ,
            world_bank=world_bank,
            hfcs=hfcs,
            eurostat=eurostat,
            ons=ons_reader,
            policy_rates=policy_rates,
            imf_reader=imf_reader,
            exchange_rates=exchange_rates,
            goods_criticality=goods_criticality,
        )


def prune_icio_dict(icio_dict: dict[int, Any], prune_date: str | int | datetime):
    # make sure prune date is the year in int format
    if isinstance(prune_date, str):
        prune_date = pd.to_datetime(prune_date, format="%Y-%m-%d").year
        prune_date = int(prune_date)
    elif isinstance(prune_date, datetime):
        prune_date = prune_date.year

    icio_dict = {year: icio for year, icio in icio_dict.items() if year >= prune_date}

    if not icio_dict:
        warnings.warn(
            f"No ICIO data was kept for date {prune_date}.",
            DataFilterWarning,
        )
    return icio_dict
