import pickle as pkl
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import time

from macro_data.configuration import DataConfiguration
from macro_data.configuration.countries import Country

from macro_data.processing.synthetic_country import SyntheticCountry
from macro_data.processing.synthetic_rest_of_the_world.default_synthetic_rest_of_the_world import (
    DefaultSyntheticRestOfTheWorld,
)
from macro_data.processing.synthetic_rest_of_the_world.synthetic_rest_of_the_world import SyntheticRestOfTheWorld
from macro_data.readers import DataReaders, compile_industry_data
from macro_data.readers.exogenous_data import ExogenousCountryData


@dataclass
class DataWrapper:
    """
    This class is used to create all the synthetic data for the INET model.
    The wrapper contains all the synthetic data needed to run the model. It contains a dictionary of synthetic
    countries, a synthetic rest of the world, exchange rates, trade proportions by origin and destination
    and a data configuration.

    Each synthetic country is itself a dataclass that contains the synthetically generated data for a single country
    and all of the agents that make it up.

    Attributes:
        synthetic_countries (dict[str, SyntheticCountry]): The synthetic countries.
        synthetic_rest_of_the_world (SyntheticRestOfTheWorld): The synthetic rest of the world.
        exchange_rates (pd.DataFrame): The exchange rates.
        origin_trade_proportions (pd.DataFrame): The trade proportions by origin.
        destination_trade_proportions (pd.DataFrame): The trade proportions by destination.
        configuration (DataConfiguration): The data configuration.
    """

    synthetic_countries: dict[str, SyntheticCountry]
    synthetic_rest_of_the_world: SyntheticRestOfTheWorld
    exchange_rates: pd.DataFrame
    origin_trade_proportions: pd.DataFrame
    destination_trade_proportions: pd.DataFrame
    configuration: DataConfiguration
    calibration_data: pd.DataFrame

    @property
    def all_country_names(self) -> list[str]:
        """
        Returns:
            list[str]: A list of all the country names.
        """
        return list(self.synthetic_countries.keys()) + ["ROW"]

    @property
    def industries(self) -> list[str]:
        """
        Returns:
            list[str]: A list of all the industry names.
        """
        return self.configuration.industries

    @property
    def n_industries(self):
        """
        Returns:
            int: The number of industries.
        """
        return len(self.industries)

    @classmethod
    def from_config(
        cls,
        configuration: DataConfiguration,
        raw_data_path: Path | str,
        single_hfcs_survey: bool = True,
        single_icio_survey: bool = True,
    ) -> "DataWrapper":
        """
        Initializes a DataWrapper object with the given parameters. The DataWrapper will contain all the synthetic data
        needed to run the model.

        Args:
            configuration (DataConfiguration): The data configuration.
            raw_data_path (Path | str): The path to the raw data.
            single_hfcs_survey (bool, optional): Whether to use a single HFCS survey. Defaults to True.
            single_icio_survey (bool, optional): Whether to use a single ICIO survey. Defaults to True.

        Returns:
            DataWrapper: The initialized DataWrapper.
        """
        # ensure that string paths are paths
        if isinstance(raw_data_path, str):
            raw_data_path = Path(raw_data_path)

        if configuration.seed is not None:
            np.random.seed(configuration.seed)
        else:
            np.random.seed(int(time.time()))

        for country, country_config in configuration.country_configs.items():
            if country_config.eu_proxy_country is None and not country.is_eu_country:
                raise ValueError(f"{country} is not in EU: please set an EU proxy country.")

        proxy_country_dict = {
            country: configuration.country_configs[country].eu_proxy_country
            for country in configuration.countries
            if not country.is_eu_country
        }

        country_names = configuration.countries
        industries = configuration.industries
        year = configuration.year
        quarter = configuration.quarter

        scale_dict = {country: configuration.country_configs[country].scale for country in country_names}

        prune_date = configuration.prune_date
        readers = DataReaders.from_raw_data(
            raw_data_path=raw_data_path,
            country_names=country_names,
            industries=industries,
            simulation_year=year,
            scale_dict=scale_dict,
            prune_date=prune_date,
            force_single_hfcs_survey=single_hfcs_survey,
            single_icio_survey=single_icio_survey,
            proxy_country_dict=proxy_country_dict,
        )

        single_firm_dict = {
            country: configuration.country_configs[country].single_firm_per_industry for country in country_names
        }

        industry_data = compile_industry_data(
            year=year, readers=readers, country_names=country_names, single_firm_per_industry=single_firm_dict
        )

        year_range = 1 if single_hfcs_survey else 10

        # exogenous_data = create_all_exogenous_data(readers, country_names, proxy_countries=proxy_country_dict)

        exogenous_data = {
            country: ExogenousCountryData.from_data_readers(
                country_name=country,
                readers=readers,
                year=year,
                quarter=quarter,
                industry_vectors=industry_data[country]["industry_vectors"],
                proxy_country=proxy_country_dict.get(country, None),
            )
            for country in country_names
        }

        proxy_inflation = {}

        non_eu_countries = [country for country in country_names if not country.is_eu_country]

        for country in non_eu_countries:
            if proxy_country_dict[country] is not None:
                proxy_country = proxy_country_dict[country]
                inflation = readers.imf_reader.get_inflation(proxy_country)
                if inflation is None:
                    inflation = readers.world_bank.get_inflation(proxy_country)
                proxy_inflation[country] = inflation
            else:
                proxy_inflation[country] = None

        calibration_data = pd.concat(
            [exogenous_data[country].get_calibration_data(year, quarter) for country in country_names], axis=1
        )

        calibration_data = add_row_to_calibration(
            calibration_data=calibration_data,
            industry_data=industry_data,
            year=year,
            quarter=quarter,
        )

        # currently only EU countries implemented

        synthetic_countries = {
            country: SyntheticCountry.eu_synthetic_country(
                country=country,
                year=year,
                quarter=quarter,
                country_configuration=configuration.country_configs[country],
                industries=industries,
                readers=readers,
                exogenous_country_data=exogenous_data[country],
                country_industry_data=industry_data[country],
                year_range=year_range,
                goods_criticality_matrix=readers.goods_criticality.criticality_matrix,
            )
            for country in country_names
            if country.is_eu_country
        }

        for country in country_names:
            if not country.is_eu_country:
                synthetic_countries[country] = SyntheticCountry.proxied_synthetic_country(
                    country=country,
                    proxy_country=configuration.country_configs[country].eu_proxy_country,
                    year=year,
                    country_configuration=configuration.country_configs[country],
                    industries=industries,
                    readers=readers,
                    exogenous_country_data=exogenous_data[country],
                    country_industry_data=industry_data[country],
                    year_range=year_range,
                    goods_criticality_matrix=readers.goods_criticality.criticality_matrix,
                    quarter=quarter,
                    proxy_inflation_data=proxy_inflation[country],
                )

        row_exports_growth = calibration_data[("ROW", "Exports (Growth)")]
        row_imports_growth = calibration_data[("ROW", "Imports (Growth)")]

        total_number_sellers = np.sum(
            [synthetic_countries[country].n_sellers_by_industry for country in synthetic_countries], axis=1
        )

        total_number_buyers = np.sum([synthetic_countries[country].n_buyers for country in synthetic_countries])

        synthetic_row = DefaultSyntheticRestOfTheWorld.from_readers(
            readers=readers,
            year=year,
            industry_data=industry_data,
            n_sellers_by_industry=total_number_sellers,
            n_buyers=total_number_buyers,
            row_configuration=configuration.row_data_config,
            row_exports_growth=row_exports_growth,
            row_imports_growth=row_imports_growth,
        )

        exchange_rates = readers.exchange_rates.df
        origin_trade_proportions = readers.icio[year].get_origin_trade_proportions()
        destination_trade_proportions = readers.icio[year].get_destination_trade_proportions()

        return cls(
            synthetic_countries=synthetic_countries,
            synthetic_rest_of_the_world=synthetic_row,
            exchange_rates=exchange_rates,
            origin_trade_proportions=origin_trade_proportions,
            destination_trade_proportions=destination_trade_proportions,
            configuration=configuration,
            calibration_data=calibration_data,
        )

    @classmethod
    def init_from_pickle(cls, path: str | Path) -> "DataWrapper":
        """
        Initialise the DataWrapper from a pickle file.

        Args:
            path (str or Path): The path to the pickle file.

        Returns:
            DataWrapper: The wrapper.
        """
        if isinstance(path, str):
            path = Path(path)

        with open(path, "rb") as f:
            data = pkl.load(f)

        return cls(**data)

    def save(self, path: str | Path) -> None:
        """
        Save the synthetic data to a pickle file.

        Args:
            path (str or Path): The path to the pickle file.
        """
        if isinstance(path, str):
            path = Path(path)

        with open(path, "wb") as f:
            pkl.dump(self.__dict__, f)

    @property
    def calibration_before(self):
        year = self.configuration.year
        quarter = self.configuration.quarter
        calibration_index = self.calibration_data.index
        calibration_before_index = calibration_index[calibration_index < f"{year}-Q{quarter}"]
        return self.calibration_data.loc[calibration_before_index]

    @property
    def calibration_during(self):
        year = self.configuration.year
        quarter = self.configuration.quarter
        calibration_index = self.calibration_data.index
        calibration_during_index = calibration_index[calibration_index == f"{year}-Q{quarter}"]
        return self.calibration_data.loc[calibration_during_index]


def add_row_to_calibration(
    calibration_data: pd.DataFrame, industry_data: dict[str | Country, dict[str, pd.DataFrame]], year: int, quarter: int
) -> pd.DataFrame:
    """
    Add the Rest of the World data to the calibration data.
    This computes the Rest of the World exports and imports based on the exports and imports of the other countries,
    along with PPI data.

    Args:
        calibration_data (pd.DataFrame): The calibration data.
        industry_data (dict[str | Country, dict[str, pd.DataFrame]]): The industry data.
        year (int): The year.
        quarter (int): The quarter.

    Returns:
        pd.DataFrame: The calibration data with the Rest of the World data added.

    """
    countries = list(calibration_data.columns.get_level_values(0).unique())

    total_country_imports_base = sum(
        [industry_data[country]["industry_vectors"]["Imports in USD"] for country in countries]
    )
    total_country_exports_base = sum(
        [industry_data[country]["industry_vectors"]["Exports in USD"] for country in countries]
    )  # type: ignore

    all_exports = calibration_data.xs("Exports (Value)", axis=1, level=1)
    all_imports = calibration_data.xs("Imports (Value)", axis=1, level=1)

    scaled_exports = all_exports / all_exports.loc[f"{year}-Q{quarter}"].iloc[0]
    scaled_imports = all_imports / all_imports.loc[f"{year}-Q{quarter}"].iloc[0]

    total_country_exports = sum(
        [country_scaled_exports(country, industry_data, scaled_exports) for country in countries]
    )

    row_scale_exports = total_country_exports / total_country_exports_base.values  # type: ignore
    row_exports_base = industry_data["ROW"]["industry_vectors"]["Exports in USD"]

    # TODO is this OK ? I'd fill with the mean
    row_scale_exports.fillna(1, inplace=True)

    row_exports = row_scale_exports * row_exports_base.values
    total_row_exports = row_exports.sum(axis=1)

    total_country_imports = sum(
        [country_scaled_imports(country, industry_data, scaled_imports) for country in countries]
    )

    row_imports = total_country_exports + row_exports - total_country_imports
    # clip to be positive imports
    total_row_imports = row_imports.sum(axis=1)

    total_nominal_output = calibration_data.xs("Gross Output (Value)", axis=1, level=1).sum(axis=1)
    total_real_output = calibration_data.xs("Real Gross Output (Value)", axis=1, level=1).sum(axis=1)

    row_ppi = total_nominal_output / total_real_output

    calibration_data[("ROW", "Exports (Value)")] = total_row_exports
    calibration_data[("ROW", "Imports (Value)")] = total_row_imports
    calibration_data[("ROW", "Exports (Growth)")] = calibration_data[("ROW", "Exports (Value)")].pct_change()
    calibration_data[("ROW", "Imports (Growth)")] = calibration_data[("ROW", "Imports (Value)")].pct_change()

    calibration_data[("ROW", "PPI (Value)")] = row_ppi
    calibration_data[("ROW", "PPI (Growth)")] = calibration_data[("ROW", "PPI (Value)")].pct_change()

    # real imports are imports over ppi
    calibration_data[("ROW", "Real Imports (Value)")] = (
        calibration_data[("ROW", "Imports (Value)")] / calibration_data[("ROW", "PPI (Value)")]
    )

    # same for exports
    calibration_data[("ROW", "Real Exports (Value)")] = (
        calibration_data[("ROW", "Exports (Value)")] / calibration_data[("ROW", "PPI (Value)")]
    )
    return calibration_data


def country_scaled_exports(
    country: str | Country, industry_data: dict[str | Country, dict[str, pd.DataFrame]], scaled_exports: pd.DataFrame
) -> pd.DataFrame:
    """
    Takes a time-series indicating the relative scaling of the exports of a country (with 1 for the base time period)
    and multiplies by the exports of the country at the base time period to get a dataframe with the scaled exports,
    with the industry numbers as columns and the time periods as rows.

    Args:
        country (str | Country): The country.
        industry_data (dict[str | Country, dict[str, pd.DataFrame]]): The industry data.
        scaled_exports (pd.DataFrame): The scaled exports.

    Returns:
        pd.DataFrame: The scaled exports.
    """
    exports = industry_data[country]["industry_vectors"]["Exports in USD"]
    scaled = scaled_exports[country]
    return pd.DataFrame(exports.values * scaled.values[:, np.newaxis], index=scaled.index).fillna(0)


def country_scaled_imports(
    country: str | Country, industry_data: dict[str | Country, dict[str, pd.DataFrame]], scaled_imports: pd.DataFrame
) -> pd.DataFrame:
    """
    Takes a time-series indicating the relative scaling of the imports of a country (with 1 for the base time period)
    and multiplies by the imports of the country at the base time period to get a dataframe with the scaled imports,
    with the industry numbers as columns and the time periods as rows.

    Args:
        country (str | Country): The country.
        industry_data (dict[str | Country, dict[str, pd.DataFrame]]): The industry data.
        scaled_imports (pd.DataFrame): The scaled imports.
    """
    imports = industry_data[country]["industry_vectors"]["Imports in USD"]
    scaled = scaled_imports[country]
    return pd.DataFrame(imports.values * scaled.values[:, np.newaxis], index=scaled.index).fillna(0)
