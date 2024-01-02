from dataclasses import dataclass
from typing import Optional

import pandas as pd

from inet_data.configuration import CountryConfiguration
from inet_data.configuration.countries import Country
from inet_data.processing import SyntheticPopulation
from inet_data.processing import (
    SyntheticPopulation,
    SyntheticFirms,
    SyntheticCreditMarket,
    SyntheticBanks,
    SyntheticCentralBank,
    SyntheticCentralGovernment,
    SyntheticGovernmentEntities,
    SyntheticHousingMarket,
    DefaultSyntheticCGovernment,
    DefaultSyntheticGovernmentEntities,
    DefaultSyntheticCentralBank,
    SyntheticHFCSPopulation,
    DefaultSyntheticFirms,
    DefaultSyntheticBanks,
    DefaultSyntheticHousingMarket,
    match_firms_with_banks,
    match_households_with_banks,
    set_housing_df,
    match_individuals_with_firms_country,
    create_firm_loan_df,
    create_household_loan_df,
    create_mortgage_loan_df,
)
from inet_data.readers import DataReaders


@dataclass
class SyntheticCountry:
    """Class for creating synthetic countries."""

    population: SyntheticPopulation
    firms: SyntheticFirms
    credit_market: SyntheticCreditMarket
    banks: SyntheticBanks
    central_bank: SyntheticCentralBank
    central_government: SyntheticCentralGovernment
    government_entities: SyntheticGovernmentEntities
    housing_market: SyntheticHousingMarket

    @classmethod
    def create_eu_synthetic_country(
        cls,
        country: Country,
        year: int,
        country_configuration: CountryConfiguration,
        industry: list[str],
        readers: DataReaders,
        exogenous_country_data: Optional[dict[str, pd.DataFrame]],
        country_industry_data: dict[str, pd.DataFrame],
        year_range: int,
    ):
        central_government = DefaultSyntheticCGovernment.from_readers(readers, country, year, year_range=year_range)

        total_unemployment_benefits = central_government.central_gov_data["Total Unemployment Benefits"].values[0]

        government_entities = DefaultSyntheticGovernmentEntities.from_readers(
            readers=readers,
            country_name=country,
            year=year,
            exogenous_country_data=exogenous_country_data,
            industry_data=country_industry_data,
            single_government_entity=country_configuration.single_government_entity,
        )

        central_bank = DefaultSyntheticCentralBank.from_readers(country, year, readers)

        population = SyntheticHFCSPopulation.from_readers(
            readers=readers,
            country_name=country,
            year=year,
            industry_data=country_industry_data,
            industries=industry,
            scale=country_configuration.scale,
            total_unemployment_benefits=total_unemployment_benefits,
            country_name_short=country.to_two_letter_code(),
        )

        firms = DefaultSyntheticFirms.from_readers(
            readers=readers,
            country_name=country,
            year=year,
            industry_data=country_industry_data,
            industries=industry,
            scale=scale,
            n_employees_per_industry=population.n_employees_per_industry,
        )
