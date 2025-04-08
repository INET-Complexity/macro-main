import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.configuration.region import Region
from macro_data.readers.criticality_data.goods_criticality_reader import (
    GoodsCriticalityReader,
)
from macro_data.readers.economic_data.ecb_reader import ECBReader
from macro_data.readers.economic_data.eurostat_reader import EuroStatReader
from macro_data.readers.economic_data.exchange_rates import ExchangeRatesReader
from macro_data.readers.economic_data.imf_reader import IMFReader
from macro_data.readers.economic_data.oecd_economic_data import OECDEconData
from macro_data.readers.economic_data.ons_reader import ONSReader
from macro_data.readers.economic_data.policy_rates import PolicyRatesReader
from macro_data.readers.economic_data.world_bank_reader import WorldBankReader
from macro_data.readers.emissions.emissions_reader import EmissionsReader
from macro_data.readers.icio_sea_matching import (
    add_investment_matrix_to_icio,
    get_investment_fractions,
    match_iot_with_sea,
    reconcile_value_added,
)
from macro_data.readers.io_tables.icio_reader import ICIOReader, split_gfcf_column
from macro_data.readers.io_tables.industries import AGGREGATED_INDUSTRIES
from macro_data.readers.io_tables.mappings import ICIO_AGGREGATE, ICIO_ALL
from macro_data.readers.population_data.compustat_banks_reader import (
    CompustatBanksReader,
)
from macro_data.readers.population_data.compustat_firms_reader import (
    CompustatFirmsReader,
)
from macro_data.readers.population_data.hfcs_reader import HFCSReader
from macro_data.readers.socioeconomic_data.wiod_sea_data import WIODSEAReader
from macro_data.readers.util.prune_util import DataFilterWarning


@dataclass
class DataPaths:
    goods_criticality_path: Path
    exchange_rates_path: Path
    eurostat_path: Path
    hfcs_path: Path
    icio_paths: dict[int, Path]
    icio_pivot_paths: dict[int, Path]
    wiod_sea_path: Path
    oecd_econ_path: Path
    oecd_econ_mapping_path: Path
    policy_rates_path: Path
    country_codes_path: Path
    imf_path: Path
    ons_path: Path
    world_bank_path: Path
    ecb_path: Path
    compustat_firms_annual_path: Path
    compustat_firms_quarterly_path: Path
    compustat_banks_path: Path
    emissions_path: Path

    @classmethod
    def default_paths(cls, raw_data_path: Path, icio_years: Iterable[int]):
        return cls(
            goods_criticality_path=raw_data_path / "ihs_markit_goods_criticality" / "UK_2020.csv",
            exchange_rates_path=raw_data_path / "exchange_rates" / "exchange_rates.csv",
            eurostat_path=raw_data_path / "eurostat",
            hfcs_path=raw_data_path / "hfcs",
            # icio_paths={year: raw_data_path / "icio" / str(year) / f"ICIO2021_{year}.csv" for year in icio_years},
            icio_paths={year: raw_data_path / "icio" / f"{year}_SML.csv" for year in icio_years},
            # icio_pivot_paths={
            #     year: raw_data_path / "icio" / str(year) / f"ICIO2021_{year}_pivot.csv" for year in icio_years
            # },
            icio_pivot_paths={year: raw_data_path / "icio" / f"{year}_SML_P.csv" for year in icio_years},
            wiod_sea_path=raw_data_path / "wiod_sea" / "wiod_sea.csv",
            oecd_econ_path=raw_data_path / "oecd_econ",
            oecd_econ_mapping_path=raw_data_path / "oecd_econ" / "mappings.json",
            policy_rates_path=raw_data_path / "policy_rates" / "bis_cb_policy_rates.csv",
            country_codes_path=raw_data_path / "notation" / "wikipedia-iso-country-codes.csv",
            imf_path=raw_data_path / "imf",
            ons_path=raw_data_path / "ons",
            world_bank_path=raw_data_path / "world_bank",
            ecb_path=raw_data_path / "ecb",
            compustat_firms_annual_path=raw_data_path / "compustat" / "firms_annual.csv",
            compustat_firms_quarterly_path=raw_data_path / "compustat" / "firms_quarterly.csv",
            compustat_banks_path=raw_data_path / "compustat" / "banks.csv",
            emissions_path=raw_data_path / "emissions",
        )

    # @classmethod
    # def all_industries(cls, raw_data_path: Path, icio_years: Iterable[int]):
    #     paths = cls.default_paths(raw_data_path, icio_years)
    #     paths.icio_agg_path = raw_data_path / "icio" / "mappings_all_industries.json"
    #     paths.wiod_sea_agg_path = raw_data_path / "wiod_sea" / "mappings_all_industries.json"
    #     return paths


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
    exchange_rates: ExchangeRatesReader
    goods_criticality: GoodsCriticalityReader
    ecb_reader: ECBReader
    compustat_firms: CompustatFirmsReader
    compustat_banks: CompustatBanksReader
    emissions: EmissionsReader
    regions_dict: Optional[dict[Country, list[Region]]] = None

    @classmethod
    def from_raw_data(
        cls,
        raw_data_path: Path | str,
        country_names: list[Country | Region],
        simulation_year: int,
        scale_dict: dict[Country, int],
        industries: list[str],
        aggregate_industries: bool = True,
        imputed_rent_year: int = 2014,
        exog_data_range: Tuple[int, int] = (2010, 2018),
        prune_date: Optional[date] = None,
        force_single_hfcs_survey: bool = False,
        single_icio_survey: bool = False,
        proxy_country_dict: dict[Country, Country] = None,
        use_disagg_can_2014_reader: bool = False,
        use_provincial_can_reader: bool = False,
        regions_dict: dict[Country, list[Region]] = None,
    ):
        if regions_dict:
            all_regions = [region for regions in regions_dict.values() for region in regions]
            country_names = list(set(country_names) - set(all_regions))

        if proxy_country_dict is None:
            proxy_country_dict = {country: country for country in country_names}

        raw_data_path = Path(raw_data_path)
        short_names = {country_name: country_name.to_two_letter_code() for country_name in country_names}

        if single_icio_survey:
            all_years = [simulation_year]
        else:
            all_years = range(exog_data_range[0], exog_data_range[1] + 1)

        datapaths = DataPaths.default_paths(raw_data_path, all_years)

        icio_mapping = ICIO_AGGREGATE if aggregate_industries else ICIO_ALL

        goods_criticality = GoodsCriticalityReader.from_csv(path=datapaths.goods_criticality_path)
        exchange_rates = ExchangeRatesReader.from_csv(path=datapaths.exchange_rates_path)
        eurostat = EuroStatReader(path=datapaths.eurostat_path, country_code_path=datapaths.country_codes_path)

        proxified = [country if country.is_eu_country else proxy_country_dict[country] for country in country_names]

        hfcs = {
            proxy_country: HFCSReader.from_csv(
                country_name=proxy_country,
                country_name_short=proxy_country.to_two_letter_code(),
                hfcs_data_path=datapaths.hfcs_path,
                year=simulation_year,
                exchange_rates=exchange_rates,
                num_surveys=1 if force_single_hfcs_survey else 5,
            )
            for country_name, proxy_country in zip(country_names, proxified)
        }

        eu_only = [country for country in country_names if country.is_eu_country]
        proxy_eu = list(proxy_country_dict.values())

        eu_only = list(set(eu_only).union(set(proxy_eu)))

        def get_investment_year(year: int, country_names_: Optional[list[Country | Region]] = None):
            if country_names_ is None:
                country_names_ = country_names
            return get_investment_fractions(country_names_, eurostat, proxy_country_dict, year)

        icio = {
            year: ICIOReader.agg_from_csv(
                path=datapaths.icio_paths[year],
                pivot_path=datapaths.icio_pivot_paths[year],
                considered_countries=country_names,
                industries=industries,
                year=year,
                exchange_rates=exchange_rates,
                imputed_rent_fraction=eurostat.get_imputed_rent_fraction(eu_only, imputed_rent_year),
                investment_fractions=get_investment_year(year),
                proxy_country_dict=proxy_country_dict,
                aggregation_type="Aggregate" if aggregate_industries else "All",
            )
            for year in all_years
        }

        if use_disagg_can_2014_reader:
            # check that only Canada is in the country names
            if country_names != [Country("CAN")]:
                raise ValueError("Only Canada is supported for this reader.")

            if simulation_year != 2014:
                raise ValueError("Only 2014 is supported for this reader.")
            disagg_path = raw_data_path / "icio" / "icio_can_2014_disagg.csv"
            df = pd.read_csv(disagg_path, header=[0, 1], index_col=[0, 1])
            icio[simulation_year].iot = df
            industries = df.loc["ROW"].index.unique()
            icio[simulation_year].industries = industries

        if use_provincial_can_reader:
            # check that Canada is in the country names
            if Country("CAN") not in country_names:
                raise ValueError("Canada must be in the country names for this reader.")
            if not regions_dict:
                raise ValueError("Must provide regional disaggregation dictionary.")
            if simulation_year != 2014:
                raise ValueError("Only 2014 is supported for this reader.")
            disagg_path = raw_data_path / "icio" / "icio_2014_can_provinces.csv"
            df = pd.read_csv(disagg_path, header=[0, 1], index_col=[0, 1])

            all_provinces = []
            for key, value in regions_dict.items():
                all_provinces.extend(value)

            countries_set = set(all_provinces).union(set(country_names)) - set(regions_dict.keys())
            # countries_set = countries_set.union(Country("ROW"))

            countries_and_regions = list(countries_set)

            # df = normalise_iot(
            #     iot=df,
            #     industries=industries,
            #     considered_countries=countries_and_regions,
            #     investment_fractions=get_investment_year(simulation_year, countries_and_regions),
            # )

            industry_cols = df.columns.get_level_values(1).isin(industries)
            non_total_rows = df.index.get_level_values(0) != "TOTAL"

            df.loc[("TOTAL", "Intermediate Inputs"), industry_cols] = df.loc[non_total_rows, industry_cols].sum(axis=0)

            df.rename(columns={"OUT": "TOTAL"}, level=0, inplace=True)

            df = split_gfcf_column(
                considered_countries=countries_and_regions,
                industries=industries,
                iot=df,
                investment_fractions=get_investment_year(simulation_year, countries_and_regions),
            )

            icio[simulation_year].iot = df.sort_index()
            icio[simulation_year].considered_countries = countries_and_regions

            # country_names = all_countries
        else:
            countries_and_regions = None

        if countries_and_regions is None:
            value_added_dict = {
                country_name: icio[simulation_year].get_value_added_series(country_name)
                * icio[simulation_year].yearly_factor
                for country_name in country_names
            }
        else:
            value_added_dict = {
                country_name: icio[simulation_year].get_value_added_series(country_name)
                * icio[simulation_year].yearly_factor
                for country_name in countries_and_regions
            }
            for key, value in regions_dict.items():
                value_added_dict[key] = sum([value_added_dict[region] for region in value])

        wiod_sea = WIODSEAReader.agg_from_csv(
            path=datapaths.wiod_sea_path,
            year=simulation_year,
            industries=industries,
            exchange_rates=exchange_rates,
            country_names=country_names,
            value_added_dict=value_added_dict,
            aggregation_type="Aggregate" if aggregate_industries else "All",
            regions_dict=regions_dict,
        )

        reconcile_value_added(
            icio_reader=icio[simulation_year],
            sea_reader=wiod_sea,
            country_names=country_names,
            regions_dict=regions_dict,
        )

        add_investment_matrix_to_icio(
            icio_reader=icio[simulation_year],
            sea_reader=wiod_sea,
            country_names=country_names,
            regions_dict=regions_dict,
        )

        match_iot_with_sea(
            icio_reader=icio[simulation_year],
            sea_reader=wiod_sea,
            country_names=country_names,
            regions_dict=regions_dict,
        )

        oecd_econ = OECDEconData(
            path=datapaths.oecd_econ_path,
            scale_dict=scale_dict,
        )

        policy_rates = PolicyRatesReader(
            path=datapaths.policy_rates_path, country_code_path=datapaths.country_codes_path
        )

        imf_reader = IMFReader.from_data(data_path=datapaths.imf_path, scale_dict=scale_dict)

        ons_reader = ONSReader(path=datapaths.ons_path)

        world_bank = WorldBankReader(path=datapaths.world_bank_path)

        ecb_reader = ECBReader(path=datapaths.ecb_path)

        all_countries = list(set(country_names).union(set(proxy_country_dict.values())))

        compustat_firms = CompustatFirmsReader.from_raw_data(
            year=simulation_year,
            quarter=1,
            countries=all_countries,
            raw_annual_path=datapaths.compustat_firms_annual_path,
            raw_quarterly_path=datapaths.compustat_firms_quarterly_path,
        )

        compustat_banks = CompustatBanksReader.from_raw_data(
            year=simulation_year, quarter=1, raw_quarterly_path=datapaths.compustat_banks_path, countries=all_countries
        )

        if prune_date:
            exchange_rates.prune(prune_date)
            eurostat.prune(prune_date)
            icio = prune_icio_dict(icio, prune_date)
            wiod_sea.prune(prune_date)
            oecd_econ.prune(prune_date)
            policy_rates.prune(prune_date)
            imf_reader.prune(prune_date)
            world_bank.prune(prune_date)

        emissions = EmissionsReader.read_price_data(datapaths.emissions_path)

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
            ecb_reader=ecb_reader,
            compustat_firms=compustat_firms,
            compustat_banks=compustat_banks,
            emissions=emissions,
            regions_dict=regions_dict,
        )

    @classmethod
    def get_investment_fractions(
        cls,
        country_names: list[Country],
        eurostat: EuroStatReader,
        proxy_country_dict: dict[Country, Country],
        year: int,
    ) -> dict[Country, dict[str, float]]:
        investment_fractions = {}
        for country_name in country_names:
            if country_name.is_eu_country:
                investment_fractions[country_name] = eurostat.get_investment_fractions_of_country(
                    country_name, year=year
                )
            else:
                investment_fractions[country_name] = eurostat.get_investment_fractions_of_country(
                    proxy_country_dict[country_name], year=year
                )
        return investment_fractions

    def get_exogenous_data(self, country_name: Country) -> Optional[dict[str, Any]]:
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
        self, country_name: Country, year_min: int, year_max: int, exogenous_data: dict[str, Any]
    ) -> pd.DataFrame:
        years = range(year_min, year_max)
        # unemp = [self.get_total_unemployment_benefits_lcu(country_name, year) for year in years]
        unemp = [
            self.oecd_econ.unemployment_benefits_gdp_pct(country_name, year)
            * self.world_bank.get_current_scaled_gdp(country_name, year)
            for year in years
        ]
        other = [
            self.oecd_econ.all_benefits_gdp_pct(country_name, year)
            * self.world_bank.get_current_scaled_gdp(country_name, year)
            - unemp[i]
            for i, year in enumerate(years)
        ]
        # other = [
        #     self.get_total_benefits_lcu(country_name, year)
        #     - self.get_total_unemployment_benefits_lcu(country_name, year)
        #     for year in years
        # ]

        benefits_data = pd.DataFrame(
            data={"Unemployment Benefits": unemp, "Other Total Benefits": other},
            index=pd.DatetimeIndex(
                pd.date_range(
                    start=f"{years[0]}-01-01",
                    end=f"{years[-1] + 1}-01-01",
                    freq="YE",
                )
            ),
        )

        benefits_data = benefits_data.resample("QE").interpolate("linear")
        benefits_data.index = pd.DatetimeIndex([pd.Timestamp(d.year, d.month, 1) for d in benefits_data.index])
        log_inflation = exogenous_data["log_inflation"]["Real CPI Inflation"].copy()
        log_inflation.index = pd.to_datetime(log_inflation.index, format="%Y-%m")
        data = pd.merge_asof(benefits_data, log_inflation, left_index=True, right_index=True)
        unemployment_rate = exogenous_data["unemployment_rate"]["Unemployment Rate"].copy()
        unemployment_rate.index = pd.to_datetime(unemployment_rate.index, format="%Y-%m")
        data = pd.merge_asof(data, unemployment_rate, left_index=True, right_index=True)
        return data

    def get_total_benefits_lcu(self, country_name: Country, year: int) -> float:
        return self.oecd_econ.all_benefits_gdp_pct(country_name, year) * self.world_bank.get_current_scaled_gdp(
            country_name, year
        )

    def get_total_unemployment_benefits_lcu(self, country_name: Country, year: int) -> float:
        return self.oecd_econ.unemployment_benefits_gdp_pct(
            country_name, year
        ) * self.world_bank.get_current_scaled_gdp(country_name, year)

    def get_govt_debt_lcu(self, country: Country, year: int) -> float:
        return self.oecd_econ.general_gov_debt(country, year) * self.exchange_rates.from_usd_to_lcu(country, year)

    def get_export_taxes(self, country: Country, year: int) -> float:
        return (
            self.world_bank.get_lcu_exports(country, year)
            * self.exchange_rates.from_usd_to_lcu(country, year)
            / self.icio[year].get_exports(country).sum()
        )

    def get_national_accounts_growth(self, country: Country) -> pd.DataFrame:
        if isinstance(country, Region):
            country = country.parent_country
        imf_growth = self.imf_reader.get_na_growth_rates(country)
        oecd_growth = self.oecd_econ.get_na_growth_rates(country)

        # pick columns of oecd growth not in imf growth
        oecd_growth = oecd_growth[oecd_growth.columns.difference(imf_growth.columns)]

        # merge the two dataframes, ensuring that imf growth has the index
        merged = pd.merge_asof(imf_growth, oecd_growth, left_index=True, right_index=True)
        merged = merged.loc[imf_growth.index]
        return merged

    def expand_weights_by_income(self, year: int, country: str | Country):

        weights_by_income = self.oecd_econ.get_household_consumption_by_income_quantile(country=country, year=year)
        weights_by_income.index = AGGREGATED_INDUSTRIES
        consumption_shares = self.icio[year].get_consumption_shares_series(country)

        weights_by_income_all = pd.DataFrame(index=consumption_shares.index, columns=weights_by_income.columns)

        dictionary = self.icio[year].get_updated_dictionary()

        for aggregate_industry in AGGREGATED_INDUSTRIES:
            sub_industries = dictionary.get(aggregate_industry, [])
            if not sub_industries:
                continue

            sub_industries = [s_ind for s_ind in sub_industries if s_ind in consumption_shares.index]

            shares = consumption_shares.loc[sub_industries]
            shares /= shares.sum()
            agg_weights = weights_by_income.loc[aggregate_industry]
            sub_weights = pd.DataFrame(
                np.outer(shares.values, agg_weights.values), index=sub_industries, columns=weights_by_income.columns
            )
            weights_by_income_all.loc[sub_industries] = sub_weights

        weights_by_income_all.index = range(weights_by_income_all.shape[0])
        weights_by_income = weights_by_income_all
        return weights_by_income


def prune_icio_dict(icio_dict: dict[int, Any], prune_date: date):
    # make sure prune date is the year in int format

    icio_dict = {year: icio for year, icio in icio_dict.items() if year >= prune_date.year}

    if not icio_dict:
        warnings.warn(
            f"No ICIO data was kept for date {prune_date}.",
            DataFilterWarning,
        )
    return icio_dict
