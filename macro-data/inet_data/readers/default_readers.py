import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Tuple, Any, Optional

import numpy as np
import pandas as pd

from inet_data.configuration.countries import Country
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
    def from_raw_data(
        cls,
        raw_data_path: Path | str,
        country_names: list[str],
        country_names_short: list[str],
        simulation_year: int,
        scale_dict: dict[Country, int],
        industries: list[str],
        imputed_rent_year: int = 2014,
        exog_data_range: Tuple[int, int] = (2010, 2018),
        prune_date: Optional[date] = None,
        force_single_hfcs_survey: bool = False,
        single_icio_survey: bool = False,
    ):
        raw_data_path = Path(raw_data_path)
        short_names = {
            country_name: country_name_short
            for country_name, country_name_short in zip(country_names, country_names_short)
        }
        if single_icio_survey:
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
            scale_dict=scale_dict,
        )

        policy_rates = PolicyRatesReader(
            path=datapaths.policy_rates_path, country_code_path=datapaths.country_codes_path
        )

        imf_reader = IMFReader.from_data(data_path=datapaths.imf_path, scale_dict=scale_dict)

        ons_reader = ONSReader(path=datapaths.ons_path)

        world_bank = WorldBankReader(path=datapaths.world_bank_path)

        if prune_date:
            exchange_rates.prune(prune_date)
            eurostat.prune(prune_date)
            icio = prune_icio_dict(icio, prune_date)
            wiod_sea.prune(prune_date)
            oecd_econ.prune(prune_date)
            policy_rates.prune(prune_date)
            imf_reader.prune(prune_date)
            world_bank.prune(prune_date)

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

    def get_exogenous_data(self, country_name: str) -> Optional[dict[str, Any]]:
        try:
            return {
                "log_inflation": self.world_bank.get_log_inflation(country_name),
                "sectoral_growth": self.eurostat.get_perc_sectoral_growth(country_name),
                "unemployment_rate": self.oecd_econ.get_unemployment_rate(country_name),
                "house_price_index": self.oecd_econ.get_house_price_index(country_name),
                "vacancy_rate": self.oecd_econ.get_vacancy_rate(country_name),
                "total_firm_deposits_and_debt": self.eurostat.get_total_industry_debt_and_deposits(country_name),
            }
        except KeyError:
            return None

    def get_benefits_inflation_data(
        self, country_name: str, year_min: int, year_max: int, exogenous_data: dict[str, Any]
    ) -> pd.DataFrame:
        years = range(year_min, year_max)
        unemp = [self.get_total_unemployment_benefits(country_name, year) for year in years]
        other = [
            self.get_total_benefits(country_name, year) - self.get_total_unemployment_benefits(country_name, year)
            for year in years
        ]

        benefits_data = pd.DataFrame(
            data={"Unemployment Benefits": unemp, "Other Total Benefits": other},
            index=pd.DatetimeIndex(
                pd.date_range(
                    start=f"{years[0]}-01-01",
                    end=f"{years[-1] + 1}-01-01",
                    freq="Y",
                )
            ),
        )

        benefits_data = benefits_data.resample("M").interpolate("linear")
        benefits_data.index = pd.DatetimeIndex([pd.Timestamp(d.year, d.month, 1) for d in benefits_data.index])
        log_inflation = exogenous_data["log_inflation"]["Real CPI Inflation"].copy()
        log_inflation.index = pd.to_datetime(log_inflation.index, format="%Y-%m")
        data = pd.merge_asof(benefits_data, log_inflation, left_index=True, right_index=True)
        unemployment_rate = exogenous_data["unemployment_rate"]["Unemployment Rate"].copy()
        unemployment_rate.index = pd.to_datetime(unemployment_rate.index, format="%Y-%m")
        data = pd.merge_asof(data, unemployment_rate, left_index=True, right_index=True)
        return data

    def get_total_benefits(self, country_name, year):
        return self.oecd_econ.all_benefits_gdp_pct(country_name, year) * self.world_bank.get_current_monthly_gdp(
            country_name, year
        )

    def get_total_unemployment_benefits(self, country_name, year):
        return self.oecd_econ.unemployment_benefits_gdp_pct(
            country_name, year
        ) * self.world_bank.get_current_monthly_gdp(country_name, year)


def prune_icio_dict(icio_dict: dict[int, Any], prune_date: date):
    # make sure prune date is the year in int format

    icio_dict = {year: icio for year, icio in icio_dict.items() if year >= prune_date.year}

    if not icio_dict:
        warnings.warn(
            f"No ICIO data was kept for date {prune_date}.",
            DataFilterWarning,
        )
    return icio_dict


def add_investment_matrix_to_icio(
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


def get_sea(
    country_name: str,
    field: str,
    sea_reader: WIODSEAReader,
) -> np.ndarray:
    return sea_reader.df.loc[
        sea_reader.df.index.get_level_values(0) == country_name,
        field,
    ].values


def match_iot_with_sea(
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
