from dataclasses import dataclass
from pathlib import Path

import numpy as np

from inet_data.processing.synthetic_banks.default_synthetic_banks import (
    SyntheticDefaultBanks,
)
from inet_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks
from inet_data.processing.synthetic_central_bank.default_synthetic_central_bank import (
    SyntheticDefaultCentralBanks,
)
from inet_data.processing.synthetic_central_bank.synthetic_central_bank import (
    SyntheticCentralBank,
)
from inet_data.processing.synthetic_central_government.default_synthetic_central_government import (
    SyntheticDefaultCentralGovernment,
)
from inet_data.processing.synthetic_central_government.synthetic_central_government import (
    SyntheticCentralGovernment,
)
from inet_data.processing.synthetic_credit_market.default_synthetic_credit_market import (
    DefaultSyntheticCreditMarket,
)
from inet_data.processing.synthetic_credit_market.synthetic_credit_market import (
    SyntheticCreditMarket,
)
from inet_data.processing.synthetic_firms.default_synthetic_firms import (
    SyntheticDefaultFirms,
)
from inet_data.processing.synthetic_firms.synthetic_firms import SyntheticFirms
from inet_data.processing.synthetic_government_entities.default_synthetic_government_entities import (
    SyntheticDefaultGovernmentEntities,
)
from inet_data.processing.synthetic_government_entities.synthetic_government_entities import (
    SyntheticGovernmentEntities,
)
from inet_data.processing.synthetic_housing_market.default_synthetic_housing_market import (
    DefaultSyntheticHousingMarket,
)
from inet_data.processing.synthetic_housing_market.synthetic_housing_market import (
    SyntheticHousingMarket,
)

from inet_data.processing.synthetic_population.hfcs_synthetic_population import (
    SyntheticHFCSPopulation,
)
from inet_data.processing.synthetic_population.synthetic_population import (
    SyntheticPopulation,
)
from inet_data.processing.synthetic_rest_of_the_world.default_synthetic_rest_of_the_world import (
    DefaultSyntheticRestOfTheWorld,
)
from inet_data.readers.default_readers import DataReaders
from inet_data.readers.util.exogenous_data import create_all_exogenous_data
from inet_data.readers.util.industry_extraction import compile_industry_data
from inet_data.util.country_code_map import get_map_long_to_short
from inet_data.util.process_config import process_config


@dataclass
class Creator:
    """
    This class is used to create all the synthetic data for the INET model.
    Dictionaries have country names as keys and the corresponding synthetic data as values.
    """

    synthetic_population: dict[str, SyntheticPopulation]
    synthetic_firms: dict[str, SyntheticFirms]
    synthetic_banks: dict[str, SyntheticBanks]
    synthetic_credit_market: dict[str, SyntheticCreditMarket]
    synthetic_central_bank: dict[str, SyntheticCentralBank]
    synthetic_central_government: dict[str, SyntheticCentralGovernment]
    synthetic_government_entities: dict[str, SyntheticGovernmentEntities]
    synthetic_housing_market: dict[str, SyntheticHousingMarket]
    synthetic_rest_of_the_world: DefaultSyntheticRestOfTheWorld

    @staticmethod
    def default_init(
        configuration: str | Path | dict,
        raw_data_path: Path | str,
        processed_data_path: Path | str,
        random_seed: int = 0,
        create_exogenous_industry_data: bool = True,
        testing: bool = True,
    ):
        # ensure that string paths are paths
        if isinstance(raw_data_path, str):
            raw_data_path = Path(raw_data_path)
        if isinstance(processed_data_path, str):
            processed_data_path = Path(processed_data_path)

        configuration = process_config(configuration)
        np.random.seed(random_seed)

        country_names = configuration["model"]["country_names"]["value"]
        country_names_short = list(map(get_map_long_to_short(raw_data_path).get, country_names))
        industries = configuration["model"]["industries"]["value"]
        scale = configuration["model"]["scale"]["value"]
        year = configuration["model"]["year"]["value"]
        single_firm_per_industry = configuration["model"]["single_firm_per_industry"]["value"]
        single_government_entity = configuration["model"]["single_government_entity"]["value"]
        assume_zero_initial_deposits = {
            country_name: configuration["init"][country_name]["firms"]["parameters"]["assume_zero_initial_deposits"][
                "value"
            ]
            for country_name in country_names
        }
        assume_zero_initial_debt = {
            country_name: configuration["init"][country_name]["firms"]["parameters"]["assume_zero_initial_debt"][
                "value"
            ]
            for country_name in country_names
        }
        single_bank = configuration["model"]["single_bank"]["value"]
        # FUNCTIONS: this is a paramter of the central gov functions
        rent_as_fraction_of_unemployment_rate = 0.25

        prune_date = configuration["model"]["prune_date"]["value"]

        readers = DataReaders.init_default_raw_data_path(
            raw_data_path=raw_data_path,
            country_names=country_names,
            country_names_short=country_names_short,
            industries=industries,
            simulation_year=year,
            scale=scale,
            prune_date=prune_date,
            create_exogenous_industry_data=create_exogenous_industry_data,
        )

        industry_data = compile_industry_data(
            year=year, readers=readers, country_names=country_names, single_firm_per_industry=single_firm_per_industry
        )

        exogenous_data = create_all_exogenous_data(readers, country_names) if create_exogenous_industry_data else None

        synthetic_central_governments = {
            country: SyntheticDefaultCentralGovernment.create_from_readers(readers, country, year)
            for country in country_names
        }

        total_unemployment_benefits = {
            country: synthetic_central_governments[country].central_gov_data["Total Unemployment Benefits"].values[0]
            for country in country_names
        }

        synthetic_gov_entities = {
            country: SyntheticDefaultGovernmentEntities.create_from_readers(
                readers=readers,
                country_name=country,
                year=year,
                exogenous_data=exogenous_data,
                industry_data=industry_data,
                single_government_entity=single_government_entity,
            )
            for country in country_names
        }

        synthetic_central_banks = {
            country: SyntheticDefaultCentralBanks.init_from_readers(country, year, readers) for country in country_names
        }

        synthetic_population = {
            country: SyntheticHFCSPopulation.create_from_readers(
                readers=readers,
                country_name=country,
                country_name_short=country_short,
                year=year,
                scale=scale,
                industries=industries,
                industry_data=industry_data,
                rent_as_fraction_of_unemployment_rate=rent_as_fraction_of_unemployment_rate,
                total_unemployment_benefits=total_unemployment_benefits[country],
            )
            for (country, country_short) in zip(country_names, country_names_short)
        }

        synthetic_firms = {
            country: SyntheticDefaultFirms.init_from_readers(
                readers=readers,
                country_name=country,
                year=year,
                industry_data=industry_data[country],
                assume_zero_initial_deposits=assume_zero_initial_deposits[country],
                assume_zero_initial_debt=assume_zero_initial_debt[country],
                industries=industries,
                n_employees_per_industry=synthetic_population[country].n_employees_per_industry,
                scale=scale,
            )
            for country in country_names
        }

        synthetic_banks = {
            country: SyntheticDefaultBanks.init_from_readers(
                single_bank=single_bank,
                country_name=country,
                year=year,
                readers=readers,
                scale=scale,
            )
            for country in country_names
        }
