import pickle as pkl
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from macro_data.configuration import DataConfiguration
from macro_data.processing.synthetic_country import SyntheticCountry
from macro_data.processing.synthetic_rest_of_the_world.default_synthetic_rest_of_the_world import (
    DefaultSyntheticRestOfTheWorld,
)
from macro_data.processing.synthetic_rest_of_the_world.synthetic_rest_of_the_world import SyntheticRestOfTheWorld
from macro_data.readers import DataReaders, compile_industry_data, create_all_exogenous_data


@dataclass
class DataWrapper:
    """
    This class is used to create all the synthetic data for the INET model.
    Consists of a dictionary with countries as keys and the synthetic countries as values.

    Attributes:
        synthetic_countries (dict[str, SyntheticCountry]): The synthetic countries.
        synthetic_rest_of_the_world (SyntheticRestOfTheWorld): The synthetic rest of the world.
        exchange_rates (pd.DataFrame): The exchange rates.
        trade_proportions (pd.DataFrame): The trade proportions.
        configuration (DataConfiguration): The data configuration.
    """

    synthetic_countries: dict[str, SyntheticCountry]
    synthetic_rest_of_the_world: SyntheticRestOfTheWorld
    exchange_rates: pd.DataFrame
    trade_proportions: pd.DataFrame
    configuration: DataConfiguration

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
        random_seed: int = 0,
        single_hfcs_survey: bool = True,
        single_icio_survey: bool = True,
    ) -> "DataWrapper":
        """
        Initializes a DataWrapper object with the given parameters. The DataWrapper will contain all the synthetic data
        needed to run the model.

        Args:
            configuration (DataConfiguration): The data configuration.
            raw_data_path (Path | str): The path to the raw data.
            random_seed (int, optional): The random seed for reproducibility. Defaults to 0.
            single_hfcs_survey (bool, optional): Whether to use a single HFCS survey. Defaults to True.
            single_icio_survey (bool, optional): Whether to use a single ICIO survey. Defaults to True.

        Returns:
            DataWrapper: The initialized DataWrapper.
        """
        # ensure that string paths are paths
        if isinstance(raw_data_path, str):
            raw_data_path = Path(raw_data_path)

        for country, country_config in configuration.country_configs.items():
            if country_config.eu_proxy_country is None and not country.is_eu_country:
                raise ValueError(f"{country} is not in EU: please set an EU proxy country.")

        proxy_country_dict = {
            country: configuration.country_configs[country].eu_proxy_country
            for country in configuration.countries
            if not country.is_eu_country
        }

        np.random.seed(random_seed)

        country_names = configuration.countries
        industries = configuration.industries
        year = configuration.year

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

        exogenous_data = create_all_exogenous_data(readers, country_names, proxy_countries=proxy_country_dict)

        # currently only EU countries implemented

        synthetic_countries = {
            country: SyntheticCountry.eu_synthetic_country(
                country=country,
                year=year,
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
                )

        synthetic_row = DefaultSyntheticRestOfTheWorld.from_readers(
            readers=readers,
            year=year,
            exogenous_row_data=exogenous_data.get("ROW", None) if exogenous_data else None,
            row_industry_data=industry_data["ROW"],
        )

        exchange_rates = readers.exchange_rates.df
        trade_proportions = readers.icio[year].get_trade_proportions()

        return cls(
            synthetic_countries=synthetic_countries,
            synthetic_rest_of_the_world=synthetic_row,
            exchange_rates=exchange_rates,
            trade_proportions=trade_proportions,
            configuration=configuration,
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
