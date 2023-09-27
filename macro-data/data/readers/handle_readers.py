import os

from tqdm import tqdm
from pathlib import Path

from data.readers.economic_data.eurostat_reader import EuroStatReader
from data.readers.economic_data.exchange_rates import WorldBankRatesReader
from data.readers.economic_data.imf_reader import IMFReader
from data.readers.economic_data.oecd_economic_data import OECDEconData
from data.readers.economic_data.ons_reader import ONSReader
from data.readers.economic_data.policy_rates import PolicyRatesReader
from data.readers.economic_data.world_bank_reader import WorldBankReader
from data.readers.criticality_data.goods_criticality_reader import (
    GoodsCriticalityReader,
)
from data.readers.io_tables.icio_reader import ICIOReader
from data.readers.population_data.hfcs_reader import HFCSReader
from data.readers.socioeconomic_data.wiod_sea_data import WIODSEAReader
from data.readers.util.matching_iot_with_sea import (
    create_investment_matrix,
    matching_iot_with_sea,
)

from typing import Any


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


def init_readers(
    raw_data_path: Path,
    country_names: list[str],
    country_names_short: list[str],
    year: int,
    scale: int,
    industries: list[str],
    create_exogenous_industry_data: bool = False,
    testing: bool = False,
) -> dict[str, Any]:
    # Goods criticality reader
    goods_criticality = GoodsCriticalityReader.from_csv(
        path=raw_data_path / "ihs_markit_goods_criticality" / "UK_2020.csv",
    )

    # Exchange rates
    exchange_rates = WorldBankRatesReader.from_csv(path=raw_data_path / "exchange_rates" / "exchange_rates.csv")

    # Eurostat data
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

    # Match IOT with SEA data
    matching_iot_with_sea(
        icio_reader=icio[year],
        sea_reader=wiod_sea,
        country_names=country_names,
    )

    # OECD data
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

    # IMF data
    imf_reader = IMFReader(path=raw_data_path / "imf", scale=scale)

    # World Bank data
    world_bank = WorldBankReader(path=raw_data_path / "world_bank")

    # ONS Firm Data
    ons_reader = ONSReader(path=raw_data_path / "ons")

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
