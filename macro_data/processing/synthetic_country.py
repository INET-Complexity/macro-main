"""
This module provides the SyntheticCountry class, which serves as a container for all synthetic
economic data related to a single country in the macroeconomic model. It handles the creation
and management of synthetic data for various economic agents and markets within a country.

The module supports:
- Creation of synthetic data for EU and non-EU countries
- Management of economic agents (firms, banks, households)
- Market simulations (credit, housing, goods)
- GDP calculations and economic indicators
- Emissions and environmental factors

Key features:
- Support for both EU countries and non-EU countries (via proxy mechanism)
- UK-specific method using WAS (Wealth and Assets Survey) data
- Integration of various economic markets and agents
- Handling of financial flows and relationships between agents
- Calculation of key economic indicators (GDP by different methods)
- Environmental impact tracking through emissions

Example:
    ```python
    from macro_data import DataConfiguration, Country
    from macro_data.processing.synthetic_country import SyntheticCountry

    # Create a synthetic EU country
    france = SyntheticCountry.eu_synthetic_country(
        country=Country.FRANCE,
        year=2023,
        quarter=1,
        country_configuration=country_config,
        industries=industries,
        readers=data_readers,
        exogenous_country_data=france_data,
        country_industry_data=industry_data,
        year_range=1,
        goods_criticality_matrix=criticality_matrix
    )

    # Create a synthetic UK country (uses WAS data)
    uk = SyntheticCountry.uk_synthetic_country(
        country=Country.UNITED_KINGDOM,
        year=2023,
        quarter=1,
        country_configuration=country_config,
        industries=industries,
        readers=data_readers,
        exogenous_country_data=uk_data,
        country_industry_data=industry_data,
        year_range=1,
        goods_criticality_matrix=criticality_matrix
    )

    # Access economic indicators
    gdp = france.gdp_output
    ```
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from macro_data.configuration import CountryDataConfiguration
from macro_data.configuration.countries import Country
from macro_data.configuration.region import Region
from macro_data.processing.country_data import TaxData
from macro_data.processing.synthetic_banks.default_synthetic_banks import (
    DefaultSyntheticBanks,
)
from macro_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks
from macro_data.processing.synthetic_central_bank.default_synthetic_central_bank import (
    DefaultSyntheticCentralBank,
)
from macro_data.processing.synthetic_central_bank.synthetic_central_bank import (
    SyntheticCentralBank,
)
from macro_data.processing.synthetic_central_government.default_synthetic_central_government import (
    DefaultSyntheticCGovernment,
)
from macro_data.processing.synthetic_central_government.synthetic_central_government import (
    SyntheticCentralGovernment,
)
from macro_data.processing.synthetic_credit_market.synthetic_credit_market import (
    SyntheticCreditMarket,
)
from macro_data.processing.synthetic_firms.default_synthetic_firms import (
    DefaultSyntheticFirms,
)
from macro_data.processing.synthetic_firms.synthetic_firms import SyntheticFirms
from macro_data.processing.synthetic_goods_market.synthetic_goods_market import (
    SyntheticGoodsMarket,
)
from macro_data.processing.synthetic_government_entities.default_synthetic_government_entities import (
    DefaultSyntheticGovernmentEntities,
)
from macro_data.processing.synthetic_government_entities.synthetic_government_entities import (
    SyntheticGovernmentEntities,
)
from macro_data.processing.synthetic_housing_market.default_synthetic_housing_market import (
    DefaultSyntheticHousingMarket,
)
from macro_data.processing.synthetic_housing_market.synthetic_housing_market import (
    SyntheticHousingMarket,
)
from macro_data.processing.synthetic_matching.matching_firms_with_banks import (
    match_firms_with_banks_optimal,
)
from macro_data.processing.synthetic_matching.matching_households_with_banks import (
    match_households_with_banks_optimal,
)
from macro_data.processing.synthetic_matching.matching_households_with_houses import (
    set_housing_df,
)
from macro_data.processing.synthetic_matching.matching_individuals_with_firms import (
    match_individuals_with_firms_country,
)
from macro_data.processing.synthetic_population.hfcs_synthetic_population import (
    SyntheticHFCSPopulation,
)
from macro_data.processing.synthetic_population.was_synthetic_population import (
    SyntheticWASPopulation,
)
from macro_data.processing.synthetic_population.synthetic_population import (
    SyntheticPopulation,
)
from macro_data.readers import AGGREGATED_INDUSTRIES, ALL_INDUSTRIES, DataReaders
from macro_data.readers.emissions.emissions_reader import EmissionsData
from macro_data.readers.exogenous_data import ExogenousCountryData


@dataclass
class SyntheticCountry:
    """
    A comprehensive container for all synthetic economic data and agents within a country.

    This class serves as the primary interface for managing synthetic economic data,
    including populations, firms, markets, and financial institutions. It provides
    methods for creating and managing synthetic data for both EU and non-EU countries,
    handling economic relationships between agents, and calculating key economic indicators.

    Attributes:
        population (SyntheticPopulation): Synthetic household and individual data
        firms (SyntheticFirms): Synthetic firm data and behavior
        credit_market (SyntheticCreditMarket): Credit market operations and state
        banks (SyntheticBanks): Banking system data and operations
        central_bank (SyntheticCentralBank): Central bank policy and operations
        central_government (SyntheticCentralGovernment): Central government fiscal policy
        government_entities (SyntheticGovernmentEntities): Government agency data
        housing_market (SyntheticHousingMarket): Housing market state and operations
        synthetic_goods_market (SyntheticGoodsMarket): Goods market transactions
        dividend_payout_ratio (float): Ratio of profits paid as dividends
        long_term_interest_rate (float): Long-term interest rate for the economy
        policy_rate_markup (float): Markup over policy rate for lending
        industry_data (dict[str, pd.DataFrame]): Industry-level economic data
        goods_criticality_matrix (pd.DataFrame): Matrix of goods dependencies
        tax_data (TaxData): Tax rates and revenue data
        exogenous_data (ExogenousCountryData): External economic factors
        scale (int): Scaling factor for synthetic agents
        country_name (Country): Country identifier
        country_configuration (CountryDataConfiguration): Country-specific settings
        industries (list[str]): List of industry sectors
        consumption_weights_by_income (pd.DataFrame): Consumption patterns by income
        emission_factors (EmissionsData): Environmental impact factors
    """

    population: SyntheticPopulation
    firms: SyntheticFirms
    credit_market: SyntheticCreditMarket
    banks: SyntheticBanks
    central_bank: SyntheticCentralBank
    central_government: SyntheticCentralGovernment
    government_entities: SyntheticGovernmentEntities
    housing_market: SyntheticHousingMarket
    synthetic_goods_market: SyntheticGoodsMarket
    dividend_payout_ratio: float
    long_term_interest_rate: float
    policy_rate_markup: float
    industry_data: dict[str, pd.DataFrame]
    goods_criticality_matrix: pd.DataFrame
    tax_data: TaxData
    exogenous_data: ExogenousCountryData
    scale: int
    country_name: Country
    country_configuration: CountryDataConfiguration
    industries: list[str]
    consumption_weights_by_income: pd.DataFrame
    emission_factors: EmissionsData

    @classmethod
    def eu_synthetic_country(
        cls,
        country: Country,
        year: int,
        quarter: int,
        country_configuration: CountryDataConfiguration,
        industries: list[str],
        readers: DataReaders,
        exogenous_country_data: ExogenousCountryData,
        country_industry_data: dict[str, pd.DataFrame],
        year_range: int,
        goods_criticality_matrix: pd.DataFrame,
        emission_factors: Optional[EmissionsData] = None,
    ) -> "SyntheticCountry":
        """
        Create a synthetic country object for a European Union member country.

        This method initializes all economic agents and markets for an EU country using
        actual EU data sources. It sets up the complete economic structure including:
        - Government institutions (central bank, government entities)
        - Financial system (banks, credit markets)
        - Real economy (firms, households, goods market)
        - Environmental factors (if emission data provided)

        Args:
            country (Country): The EU country to create synthetic data for
            year (int): Base year for data generation
            quarter (int): Base quarter for data generation
            country_configuration (CountryDataConfiguration): Country-specific settings
            industries (list[str]): List of industry sectors to model
            readers (DataReaders): Data source readers
            exogenous_country_data (ExogenousCountryData): External economic factors
            country_industry_data (dict[str, pd.DataFrame]): Industry-level data
            year_range (int): Number of years of historical data to consider
            goods_criticality_matrix (pd.DataFrame): Matrix of goods dependencies
            emission_factors (Optional[EmissionsData]): Environmental impact factors

        Returns:
            SyntheticCountry: Initialized synthetic country instance

        Note:
            This method should only be used for EU member countries. For the UK,
            use uk_synthetic_country instead. For other non-EU countries,
            use proxied_synthetic_country instead.
        """
        central_government = DefaultSyntheticCGovernment.from_readers(readers, country, year, year_range=year_range)

        total_unemployment_benefits = central_government.central_gov_data["Total Unemployment Benefits"].values[0]

        government_entities = DefaultSyntheticGovernmentEntities.from_readers(
            readers=readers,
            country_name=country,
            year=year,
            quarter=quarter,
            exogenous_country_data=exogenous_country_data,
            industry_data=country_industry_data,
            single_government_entity=country_configuration.single_government_entity,
            emission_factors=emission_factors,
        )

        central_bank = DefaultSyntheticCentralBank.from_readers(
            country, year, quarter, readers, exogenous_country_data, country_configuration.central_bank_configuration
        )

        population: SyntheticHFCSPopulation = SyntheticHFCSPopulation.from_readers(
            readers=readers,
            country_name=country,
            year=year,
            quarter=quarter,
            industry_data=country_industry_data,
            industries=industries,
            scale=country_configuration.scale,
            total_unemployment_benefits=total_unemployment_benefits,
            country_name_short=country.to_two_letter_code(),
            exogenous_data=exogenous_country_data,
        )

        firms = DefaultSyntheticFirms.from_readers(
            readers=readers,
            country_name=country,
            year=year,
            industry_data=country_industry_data,
            industries=industries,
            scale=country_configuration.scale,
            n_employees_per_industry=population.number_employees_by_industry,
            firm_configuration=country_configuration.firms_configuration,
            emission_factors=emission_factors,
        )

        banks = DefaultSyntheticBanks.from_readers(
            readers=readers,
            country_name=country,
            year=year,
            scale=country_configuration.scale,
            single_bank=country_configuration.single_bank,
            banks_data_configuration=country_configuration.banks_configuration,
            quarter=quarter,
            inflation_data=exogenous_country_data.inflation,
            proxy_eu_country=None,
        )

        synthetic_goods_market = SyntheticGoodsMarket.from_readers(
            country_name=country, year=year, quarter=quarter, readers=readers, exogenous_data=exogenous_country_data
        )

        tax_data = TaxData.from_readers(readers, country, year)

        total_imputed_rent = readers.icio[year].imputed_rents[country]

        dividend_payout_ratio = readers.eurostat.dividend_payout_ratio(country=country, year=year)
        long_term_interest_rate = readers.oecd_econ.read_long_term_interest_rates(country=country, year=year)
        policy_rate_markup = readers.eurostat.firm_risk_premium(country=country, year=year)

        if set(industries).issubset(ALL_INDUSTRIES):
            weights_by_income = readers.expand_weights_by_income(year=year, country=country)
        else:
            weights_by_income = readers.oecd_econ.get_household_consumption_by_income_quantile(
                country=country, year=year
            )

        cls.match_households_firms_banks(banks, firms, industries, population, tax_data)

        housing_data = set_housing_df(
            synthetic_population=population,
            rental_income_taxes=tax_data.income_tax,
            social_housing_rent=population.social_housing_rent,
            total_imputed_rent=total_imputed_rent,
        )

        housing_market = DefaultSyntheticHousingMarket(country, housing_data)

        if emission_factors is not None:
            emitting_industry_indices = np.array(
                [list(industries).index(industry) for industry in ["B05a", "B05b", "B05c", "C19"]]
            )
            emission_factors_array = emission_factors.emissions_array
        else:
            emitting_industry_indices = None
            emission_factors_array = None

        credit_market = cls.set_wealth_and_credit(
            banks=banks,
            central_government=central_government,
            country_configuration=country_configuration,
            country_industry_data=country_industry_data,
            firms=firms,
            population=population,
            tax_data=tax_data,
            central_bank=central_bank,
            weights_by_income=weights_by_income,
            emitting_indices=emitting_industry_indices,
            emission_factors_array=emission_factors_array,
        )

        return cls(
            population=population,
            firms=firms,
            credit_market=credit_market,
            banks=banks,
            central_bank=central_bank,
            central_government=central_government,
            government_entities=government_entities,
            housing_market=housing_market,
            dividend_payout_ratio=dividend_payout_ratio,
            long_term_interest_rate=long_term_interest_rate,
            policy_rate_markup=policy_rate_markup,
            industry_data=country_industry_data,
            goods_criticality_matrix=goods_criticality_matrix,
            tax_data=tax_data,
            exogenous_data=exogenous_country_data,
            scale=country_configuration.scale,
            country_name=country,
            country_configuration=country_configuration,
            industries=industries,
            consumption_weights_by_income=weights_by_income,
            synthetic_goods_market=synthetic_goods_market,
            emission_factors=emission_factors,
        )

    @classmethod
    def uk_synthetic_country(
        cls,
        country: Country,
        year: int,
        quarter: int,
        country_configuration: CountryDataConfiguration,
        industries: list[str],
        readers: DataReaders,
        exogenous_country_data: ExogenousCountryData,
        country_industry_data: dict[str, pd.DataFrame],
        year_range: int,
        goods_criticality_matrix: pd.DataFrame,
        emission_factors: Optional[EmissionsData] = None,
    ) -> "SyntheticCountry":
        """
        Create a synthetic country object for the United Kingdom.

        This method initializes all economic agents and markets for the UK using
        UK-specific data sources. It prioritizes WAS (Wealth and Assets Survey) data
        for population, falling back to HFCS if WAS is not available. It sets up the
        complete economic structure including:
        - Government institutions (central bank, government entities)
        - Financial system (banks, credit markets)
        - Real economy (firms, households, goods market)
        - Environmental factors (if emission data provided)

        Args:
            country (Country): The UK country identifier (should be Country.UNITED_KINGDOM)
            year (int): Base year for data generation
            quarter (int): Base quarter for data generation
            country_configuration (CountryDataConfiguration): Country-specific settings
            industries (list[str]): List of industry sectors to model
            readers (DataReaders): Data source readers
            exogenous_country_data (ExogenousCountryData): External economic factors
            country_industry_data (dict[str, pd.DataFrame]): Industry-level data
            year_range (int): Number of years of historical data to consider
            goods_criticality_matrix (pd.DataFrame): Matrix of goods dependencies
            emission_factors (Optional[EmissionsData]): Environmental impact factors

        Returns:
            SyntheticCountry: Initialized synthetic country instance

        Note:
            This method is specifically designed for the UK and uses WAS data when available.
            For other EU countries, use eu_synthetic_country instead.
        """
        central_government = DefaultSyntheticCGovernment.from_readers(readers, country, year, year_range=year_range)

        total_unemployment_benefits = central_government.central_gov_data["Total Unemployment Benefits"].values[0]

        government_entities = DefaultSyntheticGovernmentEntities.from_readers(
            readers=readers,
            country_name=country,
            year=year,
            quarter=quarter,
            exogenous_country_data=exogenous_country_data,
            industry_data=country_industry_data,
            single_government_entity=country_configuration.single_government_entity,
            emission_factors=emission_factors,
        )

        central_bank = DefaultSyntheticCentralBank.from_readers(
            country, year, quarter, readers, exogenous_country_data, country_configuration.central_bank_configuration
        )

        # Try to find WAS data for UK population
        was_reader = readers.was.get(country)
        # If not found, try to find any Country object in the dictionary that represents UK
        if was_reader is None:
            for country_key in readers.was.keys():
                if (hasattr(country_key, 'value') and country_key.value == "GBR") or \
                   (hasattr(country_key, 'to_two_letter_code') and country_key.to_two_letter_code() == "GB"):
                    was_reader = readers.was[country_key]
                    break
        
        # Use WAS population if available, otherwise fall back to HFCS
        if was_reader is not None:
            population: SyntheticWASPopulation = SyntheticWASPopulation.from_readers(
                readers=readers,
                country_name=country,
                year=year,
                quarter=quarter,
                industry_data=country_industry_data,
                industries=industries,
                scale=country_configuration.scale,
                total_unemployment_benefits=total_unemployment_benefits,
                country_name_short=country.to_two_letter_code(),
                exogenous_data=exogenous_country_data,
            )
        else:
            population: SyntheticHFCSPopulation = SyntheticHFCSPopulation.from_readers(
                readers=readers,
                country_name=country,
                year=year,
                quarter=quarter,
                industry_data=country_industry_data,
                industries=industries,
                scale=country_configuration.scale,
                total_unemployment_benefits=total_unemployment_benefits,
                country_name_short=country.to_two_letter_code(),
                exogenous_data=exogenous_country_data,
            )

        firms = DefaultSyntheticFirms.from_readers(
            readers=readers,
            country_name=country,
            year=year,
            industry_data=country_industry_data,
            industries=industries,
            scale=country_configuration.scale,
            n_employees_per_industry=population.number_employees_by_industry,
            firm_configuration=country_configuration.firms_configuration,
            emission_factors=emission_factors,
        )

        # For UK (non-EU country), use a proxy EU country for Eurostat data
        proxy_eu_country = None
        if country_configuration.eu_proxy_country:
            proxy_eu_country = country_configuration.eu_proxy_country
        else:
            # Default to France if no proxy specified
            proxy_eu_country = Country.FRANCE
        
        banks = DefaultSyntheticBanks.from_readers(
            readers=readers,
            country_name=country,
            year=year,
            scale=country_configuration.scale,
            single_bank=country_configuration.single_bank,
            banks_data_configuration=country_configuration.banks_configuration,
            quarter=quarter,
            inflation_data=exogenous_country_data.inflation,
            proxy_eu_country=proxy_eu_country,
        )

        synthetic_goods_market = SyntheticGoodsMarket.from_readers(
            country_name=country, year=year, quarter=quarter, readers=readers, exogenous_data=exogenous_country_data
        )

        tax_data = TaxData.from_readers(readers, country, year)

        total_imputed_rent = readers.icio[year].imputed_rents[country]

        dividend_payout_ratio = readers.eurostat.dividend_payout_ratio(country=country, year=year)
        long_term_interest_rate = readers.oecd_econ.read_long_term_interest_rates(country=country, year=year)
        policy_rate_markup = readers.eurostat.firm_risk_premium(country=country, year=year)

        if set(industries).issubset(ALL_INDUSTRIES):
            weights_by_income = readers.expand_weights_by_income(year=year, country=country)
        else:
            weights_by_income = readers.oecd_econ.get_household_consumption_by_income_quantile(
                country=country, year=year
            )

        cls.match_households_firms_banks(banks, firms, industries, population, tax_data)

        housing_data = set_housing_df(
            synthetic_population=population,
            rental_income_taxes=tax_data.income_tax,
            social_housing_rent=population.social_housing_rent,
            total_imputed_rent=total_imputed_rent,
        )

        housing_market = DefaultSyntheticHousingMarket(country, housing_data)

        if emission_factors is not None:
            emitting_industry_indices = np.array(
                [list(industries).index(industry) for industry in ["B05a", "B05b", "B05c", "C19"]]
            )
            emission_factors_array = emission_factors.emissions_array
        else:
            emitting_industry_indices = None
            emission_factors_array = None

        credit_market = cls.set_wealth_and_credit(
            banks=banks,
            central_government=central_government,
            country_configuration=country_configuration,
            country_industry_data=country_industry_data,
            firms=firms,
            population=population,
            tax_data=tax_data,
            central_bank=central_bank,
            weights_by_income=weights_by_income,
            emitting_indices=emitting_industry_indices,
            emission_factors_array=emission_factors_array,
        )

        return cls(
            population=population,
            firms=firms,
            credit_market=credit_market,
            banks=banks,
            central_bank=central_bank,
            central_government=central_government,
            government_entities=government_entities,
            housing_market=housing_market,
            dividend_payout_ratio=dividend_payout_ratio,
            long_term_interest_rate=long_term_interest_rate,
            policy_rate_markup=policy_rate_markup,
            industry_data=country_industry_data,
            goods_criticality_matrix=goods_criticality_matrix,
            tax_data=tax_data,
            exogenous_data=exogenous_country_data,
            scale=country_configuration.scale,
            country_name=country,
            country_configuration=country_configuration,
            industries=industries,
            consumption_weights_by_income=weights_by_income,
            synthetic_goods_market=synthetic_goods_market,
            emission_factors=emission_factors,
        )

    @classmethod
    def proxied_synthetic_country(
        cls,
        country: Country | Region,
        proxy_country: Country,
        year: int,
        quarter: int,
        country_configuration: CountryDataConfiguration,
        industries: list[str],
        readers: DataReaders,
        exogenous_country_data: ExogenousCountryData,
        country_industry_data: dict[str, pd.DataFrame],
        year_range: int,
        goods_criticality_matrix: pd.DataFrame,
        proxy_inflation_data: pd.DataFrame,
        emission_factors: Optional[EmissionsData] = None,
    ) -> "SyntheticCountry":
        """
        Create a synthetic country object for a non-EU country using an EU country as proxy.

        This method creates synthetic data for non-EU countries by using an EU country's
        data structure as a template, while maintaining the non-EU country's actual:
        - Population ratios
        - Exchange rates
        - Economic scale
        - Industry structure
        - Trade patterns

        Args:
            country (Country): The non-EU country to create synthetic data for
            proxy_country (Country): The EU country to use as a template
            year (int): Base year for data generation
            quarter (int): Base quarter for data generation
            country_configuration (CountryDataConfiguration): Country-specific settings
            industries (list[str]): List of industry sectors to model
            readers (DataReaders): Data source readers
            exogenous_country_data (ExogenousCountryData): External economic factors
            country_industry_data (dict[str, pd.DataFrame]): Industry-level data
            year_range (int): Number of years of historical data to consider
            goods_criticality_matrix (pd.DataFrame): Matrix of goods dependencies
            proxy_inflation_data (pd.DataFrame): Inflation data from proxy country
            emission_factors (Optional[EmissionsData]): Environmental impact factors

        Returns:
            SyntheticCountry: Initialized synthetic country instance

        Note:
            This method should not be used for the UK. For the UK, use uk_synthetic_country instead.
        """
        central_government = DefaultSyntheticCGovernment.from_readers(readers, country, year, year_range=year_range)

        total_unemployment_benefits = central_government.central_gov_data["Total Unemployment Benefits"].values[0]

        government_entities = DefaultSyntheticGovernmentEntities.from_readers(
            readers=readers,
            country_name=country,
            year=year,
            exogenous_country_data=exogenous_country_data,
            industry_data=country_industry_data,
            single_government_entity=country_configuration.single_government_entity,
            quarter=quarter,
            emission_factors=emission_factors,
        )

        central_bank = DefaultSyntheticCentralBank.from_readers(
            country, year, quarter, readers, exogenous_country_data, country_configuration.central_bank_configuration
        )

        population_ratio = readers.world_bank.get_population(
            country=country, year=year
        ) / readers.world_bank.get_population(country=proxy_country, year=year)

        exch_rate_proxy_to_lcu = readers.exchange_rates.from_eur_to_lcu(country, year)

        population: SyntheticHFCSPopulation = SyntheticHFCSPopulation.from_readers(
            readers=readers,
            country_name=proxy_country,
            year=year,
            industry_data=country_industry_data,
            industries=industries,
            scale=country_configuration.scale,
            total_unemployment_benefits=total_unemployment_benefits,
            country_name_short=proxy_country.to_two_letter_code(),
            population_ratio=population_ratio,
            exch_rate=exch_rate_proxy_to_lcu,
            proxied_country=country,
            quarter=quarter,
            exogenous_data=exogenous_country_data,
        )

        firms = DefaultSyntheticFirms.from_readers(
            readers=readers,
            country_name=country,
            year=year,
            industry_data=country_industry_data,
            industries=industries,
            scale=country_configuration.scale,
            n_employees_per_industry=population.number_employees_by_industry,
            firm_configuration=country_configuration.firms_configuration,
            exchange_rate_from_eur=exch_rate_proxy_to_lcu,
            emission_factors=emission_factors,
        )

        banks = DefaultSyntheticBanks.from_readers(
            readers=readers,
            country_name=country,
            year=year,
            scale=country_configuration.scale,
            single_bank=country_configuration.single_bank,
            banks_data_configuration=country_configuration.banks_configuration,
            quarter=quarter,
            inflation_data=proxy_inflation_data,
            proxy_eu_country=proxy_country,
        )

        synthetic_goods_market = SyntheticGoodsMarket.from_readers(
            country_name=country, year=year, quarter=quarter, readers=readers, exogenous_data=exogenous_country_data
        )

        tax_data = TaxData.from_readers(readers, country, year)

        total_imputed_rent = readers.icio[year].imputed_rents[country]

        dividend_payout_ratio = readers.eurostat.dividend_payout_ratio(country=country, year=year)
        long_term_interest_rate = readers.oecd_econ.read_long_term_interest_rates(country=country, year=year)
        policy_rate_markup = readers.eurostat.firm_risk_premium(country=country, year=year)

        if set(industries).issubset(AGGREGATED_INDUSTRIES):
            weights_by_income = readers.oecd_econ.get_household_consumption_by_income_quantile(
                country=country, year=year
            )
        else:
            weights_by_income = readers.expand_weights_by_income(year=year, country=country)

        cls.match_households_firms_banks(banks, firms, industries, population, tax_data)

        housing_data = set_housing_df(
            synthetic_population=population,
            rental_income_taxes=tax_data.income_tax,
            social_housing_rent=population.social_housing_rent,
            total_imputed_rent=total_imputed_rent,
        )

        housing_market = DefaultSyntheticHousingMarket(country, housing_data)

        if emission_factors is not None:
            emitting_industry_indices = np.array(
                [list(industries).index(industry) for industry in ["B05a", "B05b", "B05c", "C19"]]
            )
            emission_factors_array = emission_factors.emissions_array
        else:
            emitting_industry_indices = None
            emission_factors_array = None

        credit_market = cls.set_wealth_and_credit(
            banks=banks,
            central_government=central_government,
            country_configuration=country_configuration,
            country_industry_data=country_industry_data,
            firms=firms,
            population=population,
            tax_data=tax_data,
            central_bank=central_bank,
            weights_by_income=weights_by_income,
            emitting_indices=emitting_industry_indices,
            emission_factors_array=emission_factors_array,
        )

        return cls(
            population=population,
            firms=firms,
            credit_market=credit_market,
            banks=banks,
            central_bank=central_bank,
            central_government=central_government,
            government_entities=government_entities,
            housing_market=housing_market,
            dividend_payout_ratio=dividend_payout_ratio,
            long_term_interest_rate=long_term_interest_rate,
            policy_rate_markup=policy_rate_markup,
            industry_data=country_industry_data,
            goods_criticality_matrix=goods_criticality_matrix,
            tax_data=tax_data,
            exogenous_data=exogenous_country_data,
            scale=country_configuration.scale,
            country_name=country,
            country_configuration=country_configuration,
            industries=industries,
            consumption_weights_by_income=weights_by_income,
            synthetic_goods_market=synthetic_goods_market,
            emission_factors=emission_factors,
        )

    @classmethod
    def set_wealth_and_credit(
        cls,
        banks: SyntheticBanks,
        central_government: SyntheticCentralGovernment,
        country_configuration: CountryDataConfiguration,
        country_industry_data: dict[str, pd.DataFrame],
        firms: SyntheticFirms,
        population: SyntheticPopulation,
        tax_data: TaxData,
        central_bank: SyntheticCentralBank,
        weights_by_income: pd.DataFrame,
        emission_factors_array: Optional[np.ndarray] = None,
        emitting_indices: Optional[np.ndarray] = None,
    ) -> SyntheticCreditMarket:
        """
        This function takes care of matching the different agents together and initialising the Credit
        and Housing markets.
        This function is separated because we may want to change the initialisation of firm parameters
        in particular those which depend on function parameters), in which case we need to redo the matching
        and initialisation of the markets.

        Args:
            banks (SyntheticBanks): The synthetic banks.
            central_government (SyntheticCentralGovernment): The synthetic central government.
            country_configuration (CountryDataConfiguration): The configuration data for the country.
            country_industry_data (dict[str, pd.DataFrame]): The industry data for the country.
            firms (SyntheticFirms): The synthetic firms.
            population (SyntheticPopulation): The synthetic population.
            tax_data (TaxData): The tax data for the country.
            central_bank (SyntheticCentralBank): The synthetic central bank.
            weights_by_income (pd.DataFrame): The weights by income for the country.
            emission_factors_array (np.ndarray): The emission factors for the country (tCO2 per LCU).
            emitting_indices (np.ndarray): The indices of emitting industries.

        Returns:
            tuple[SyntheticCreditMarket, SyntheticHousingMarket]: A tuple containing the synthetic credit market,
            exogenous data, and synthetic housing market.
        """

        independents = None
        # here this only changes if we change the independents of the function (e.g. income, debt)
        # not worth it to change it now

        policy_rate = central_bank.central_bank_data["policy_rate"].values[0]

        cls.initialise_pop_wealth_income(
            banks=banks,
            central_government=central_government,
            country_industry_data=country_industry_data,
            firms=firms,
            population=population,
            tax_data=tax_data,
            weights_by_income=weights_by_income,
            independents=independents,
            emission_factors_array=emission_factors_array,
            emitting_industry_indices=emitting_indices,
        )

        credit_market = cls.init_credit_market(
            banks=banks,
            central_government=central_government,
            country_configuration=country_configuration,
            country_industry_data=country_industry_data,
            firms=firms,
            population=population,
            tax_data=tax_data,
            risk_premium=tax_data.risk_premium,
            policy_rate=policy_rate,
        )
        return credit_market

    @classmethod
    def match_households_firms_banks(
        cls,
        banks: SyntheticBanks,
        firms: SyntheticFirms,
        industries: list[str],
        population: SyntheticPopulation,
        tax_data: TaxData,
        independents: Optional[list[str]] = None,
    ):
        """
        Match economic agents (households, firms, banks) to establish relationships.

        This method:
        1. Matches individuals with firms (employment relationships)
        2. Matches firms with banks (banking relationships)
        3. Computes household wealth
        4. Matches households with banks (banking relationships)

        Args:
            banks (SyntheticBanks): Banking system data
            firms (SyntheticFirms): Firm data
            industries (list[str]): List of industries
            population (SyntheticPopulation): Population data
            tax_data (TaxData): Tax rates and data
            independents (Optional[list[str]]): List of independent variables for wealth computation
        """
        income_taxes = tax_data.income_tax
        employee_social_contribution_taxes = tax_data.employee_social_insurance_tax
        match_individuals_with_firms_country(
            industries=industries,
            income_taxes=income_taxes,
            employee_social_contribution_taxes=employee_social_contribution_taxes,
            firms=firms,
            population=population,
        )
        match_firms_with_banks_optimal(firms=firms, banks=banks)
        population.compute_household_wealth(independents=independents)
        match_households_with_banks_optimal(population=population, banks=banks)

    @classmethod
    def init_credit_market(
        cls,
        banks: SyntheticBanks,
        central_government: SyntheticCentralGovernment,
        country_configuration: CountryDataConfiguration,
        country_industry_data: dict[str, pd.DataFrame],
        firms: SyntheticFirms,
        population: SyntheticPopulation,
        tax_data: TaxData,
        risk_premium: float,
        policy_rate: float,
    ) -> SyntheticCreditMarket:
        """
        Initialize the credit market for the synthetic country.

        This method sets up the credit market by:
        1. Initializing bank interest rates and profits
        2. Creating credit market relationships between agents
        3. Setting up loan installments for households
        4. Initializing firm financial conditions
        5. Updating government financial relationships

        Args:
            banks (SyntheticBanks): Banking system data
            central_government (SyntheticCentralGovernment): Government data
            country_configuration (CountryDataConfiguration): Country settings
            country_industry_data (dict[str, pd.DataFrame]): Industry data
            firms (SyntheticFirms): Firm data
            population (SyntheticPopulation): Population data
            tax_data (TaxData): Tax rates and data
            risk_premium (float): Risk premium for interest rates
            policy_rate (float): Central bank policy rate

        Returns:
            SyntheticCreditMarket: Initialized credit market
        """
        tau_bank = tax_data.profit_tax

        banks.initialise_rates_profits_liabilities(
            policy_rate=policy_rate,
            tau_bank=tau_bank,
            risk_premium=risk_premium,
            **country_configuration.banks_configuration.interest_rates.model_dump(),
        )
        credit_market = SyntheticCreditMarket.create_from_agents(
            firms=firms,
            population=population,
            banks=banks,
            firm_loan_maturity=country_configuration.banks_configuration.long_term_firm_loan_maturity,
            hh_consumption_maturity=country_configuration.banks_configuration.consumption_exp_loan_maturity,
            mortgage_maturity=country_configuration.banks_configuration.mortgage_maturity,
            zero_firm_debt=country_configuration.firms_configuration.zero_initial_debt,
        )
        population.set_debt_installments(
            consumption_installments=credit_market.consumption_expansion_loans.installments,
            mortgage_installments=credit_market.mortgage_loans.installments,
            ce_installments=credit_market.payday_loans.installments,
        )
        firms.set_additional_initial_conditions(
            tax_data=tax_data,
            industry_data=country_industry_data,
            synthetic_banks=banks,
            long_term_loans=credit_market.longterm_loans,
            short_term_loans=credit_market.shortterm_loans,
        )
        central_government.update_fields(
            synthetic_banks=banks,
            synthetic_population=population,
            synthetic_firms=firms,
            industry_data=country_industry_data,
            tax_data=tax_data,
        )
        return credit_market

    @property
    def n_sellers_by_industry(self):
        """
        Get the number of firms (sellers) in each industry.

        Returns:
            np.ndarray: Array containing the count of firms per industry
        """
        return self.firms.number_of_firms_by_industry

    @property
    def n_buyers(self):
        """
        Get the total number of economic agents that can act as buyers.

        This includes households, firms, and government entities.

        Returns:
            int: Total number of potential buyers in the economy
        """
        return (
            self.population.number_of_households
            + self.firms.number_of_firms
            + self.government_entities.number_of_entities
        )

    @classmethod
    def initialise_pop_wealth_income(
        cls,
        banks: SyntheticBanks,
        central_government: SyntheticCentralGovernment,
        country_industry_data: dict[str, pd.DataFrame],
        firms: SyntheticFirms,
        population: SyntheticPopulation,
        tax_data: TaxData,
        weights_by_income: pd.DataFrame,
        independents: Optional[list[str]] = None,
        emission_factors_array: Optional[np.ndarray] = None,
        emitting_industry_indices: Optional[np.ndarray] = None,
    ):
        """
        Initialize population wealth, income, and consumption patterns.

        This method sets up:
        1. Wealth distribution functions
        2. Household income including social transfers
        3. Saving and investment rates
        4. Consumption normalization and patterns
        5. Emissions data (if applicable)
        6. Bank deposits and loans

        Args:
            banks (SyntheticBanks): Banking system data
            central_government (SyntheticCentralGovernment): Government data
            country_industry_data (dict[str, pd.DataFrame]): Industry-level data
            firms (SyntheticFirms): Firm data
            population (SyntheticPopulation): Population data
            tax_data (TaxData): Tax rates and data
            weights_by_income (pd.DataFrame): Consumption weights by income level
            independents (Optional[list[str]]): Independent variables for wealth computation
            emission_factors_array (Optional[np.ndarray]): Emission factors by industry
            emitting_industry_indices (Optional[np.ndarray]): Indices of emitting industries
        """
        population.set_wealth_distribution_function(independents=independents)

        population.compute_household_income(
            total_social_transfers=central_government.central_gov_data["Other Social Benefits"].values[0],
            independents=independents,
        )
        population.set_household_saving_rates(independents=independents)

        population.set_household_investment_rates(capital_formation_taxrate=tax_data.capital_formation_tax)
        iot_consumption = country_industry_data["industry_vectors"]["Household Consumption in LCU"]
        population.normalise_household_consumption(
            iot_hh_consumption=iot_consumption, vat=tax_data.value_added_tax, independents=independents
        )

        population.normalise_household_investment(
            tau_cf=tax_data.capital_formation_tax,
            iot_hh_investment=country_industry_data["industry_vectors"]["Household Capital Inputs in LCU"],
        )

        population.match_consumption_weights_by_income(
            weights_by_income=weights_by_income, iot_hh_consumption=iot_consumption, vat=tax_data.value_added_tax
        )

        if (emission_factors_array is not None) and (emitting_industry_indices is not None):
            population.add_emissions(
                emission_factors_array, emitting_industry_indices, tau_cf=tax_data.capital_formation_tax
            )

        banks.initialise_deposits_and_loans(
            synthetic_population=population,
            firm_deposits=firms.firm_data["Deposits"].values,
            firm_debt=firms.firm_data["Debt"].values,
        )

    def reset_firm_function_dependent(
        self,
        capital_inputs_utilisation_rate: float,
        initial_inventory_to_input_fraction: float,
        intermediate_inputs_utilisation_rate: float,
        zero_initial_debt: bool,
        zero_initial_deposits: bool,
    ):
        """
        Reset firm parameters and reinitialize dependent markets.

        This method updates firm operational parameters and reinitializes all markets
        and relationships that depend on firm behavior, including:
        1. Firm operational parameters
        2. Housing market relationships
        3. Credit market relationships
        4. Financial flows between agents

        Args:
            capital_inputs_utilisation_rate (float): Rate of capital input usage
            initial_inventory_to_input_fraction (float): Initial inventory ratio
            intermediate_inputs_utilisation_rate (float): Rate of intermediate input usage
            zero_initial_debt (bool): Whether to reset firm debt to zero
            zero_initial_deposits (bool): Whether to reset firm deposits to zero
        """
        self.firms.reset_function_parameters(
            capital_inputs_utilisation_rate=capital_inputs_utilisation_rate,
            initial_inventory_to_input_fraction=initial_inventory_to_input_fraction,
            intermediate_inputs_utilisation_rate=intermediate_inputs_utilisation_rate,
            zero_initial_debt=zero_initial_debt,
            zero_initial_deposits=zero_initial_deposits,
        )

        housing_data = self.housing_market.housing_market_data
        owned_houses = housing_data["Is Owner-Occupied"]
        total_rent = housing_data.loc[owned_houses, "Rent"].sum()

        self.match_households_firms_banks(self.banks, self.firms, self.industries, self.population, self.tax_data)

        housing_data = set_housing_df(
            synthetic_population=self.population,
            rental_income_taxes=self.tax_data.income_tax,
            social_housing_rent=self.population.social_housing_rent,
            total_imputed_rent=total_rent,
        )

        housing_market = DefaultSyntheticHousingMarket(self.country_name, housing_data)

        credit_market = self.set_wealth_and_credit(
            banks=self.banks,
            central_government=self.central_government,
            country_configuration=self.country_configuration,
            country_industry_data=self.industry_data,
            firms=self.firms,
            population=self.population,
            tax_data=self.tax_data,
            central_bank=self.central_bank,
            weights_by_income=self.consumption_weights_by_income,
        )

        self.credit_market = credit_market
        self.housing_market = housing_market

    @property
    def gdp_output(self) -> float:
        """
        Calculate GDP using the production (output) approach.

        This method computes GDP by:
        1. Taking total sales value (production * price)
        2. Subtracting intermediate input costs
        3. Adding taxes on products
        4. Subtracting taxes on production
        5. Adding rent (both paid and imputed)

        Returns:
            float: GDP value calculated using the output approach
        """
        total_sales = (self.firms.firm_data["Production"] * self.firms.firm_data["Price"]).sum()
        used_intermediate_inputs = self.firms.used_intermediate_inputs
        used_intermediate_inputs_costs = np.matmul(self.firms.firm_data["Price"].values, used_intermediate_inputs).sum()

        total_taxes_on_products = self.central_government.central_gov_data["Taxes on Products"].values[0]
        total_taxes_on_production = self.central_government.central_gov_data["Taxes on Production"].values[0]

        rent = self.population.household_data["Rent Paid"].sum()
        imputed_rent = self.population.household_data["Rent Imputed"].sum()

        return (
            total_sales
            - used_intermediate_inputs_costs
            + total_taxes_on_products
            - total_taxes_on_production
            + rent
            + imputed_rent
        )

    @property
    def gdp_expenditure(self) -> float:
        """
        Calculate GDP using the expenditure approach.

        This method computes GDP as the sum of:
        1. Capital formation (business + household investment)
        2. Household consumption
        3. Government consumption
        4. Net exports (exports - imports)
        5. Rent (both paid and imputed)

        Returns:
            float: GDP value calculated using the expenditure approach
        """
        used_capital_inputs = self.firms.used_capital_inputs
        used_capital_inputs_costs = np.matmul(used_capital_inputs.T, self.firms.firm_data["Price"].values).sum()

        investment_rate = self.population.household_data["Investment Rate"].values
        investment_weights = self.industry_data["industry_vectors"]["Household Capital Inputs in LCU"]
        investment_weights = investment_weights.values / investment_weights.values.sum()

        income = self.population.household_data["Income"].values

        gross_hh_investment = np.outer(investment_weights, investment_rate * income).T

        capital_formation = used_capital_inputs_costs + gross_hh_investment.sum()

        hh_consumption = self.industry_data["industry_vectors"]["Household Consumption in LCU"].sum() * (
            1 + self.tax_data.value_added_tax
        )

        gov_consumption = self.government_entities.gov_entity_data["Consumption in LCU"].sum()

        exports = self.industry_data["industry_vectors"]["Exports in LCU"].sum() * (1 + self.tax_data.export_tax)

        imports = self.industry_data["industry_vectors"]["Imports in LCU"].sum()

        rent = self.population.household_data["Rent Paid"].sum()
        imputed_rent = self.population.household_data["Rent Imputed"].sum()

        return capital_formation + hh_consumption + gov_consumption + exports - imports + rent + imputed_rent

    @property
    def gdp_income(self) -> 0:
        """
        Calculate GDP using the income approach.

        This method computes GDP as the sum of:
        1. Operating surplus (sales - wages - intermediate inputs - production taxes)
        2. Wages
        3. Taxes on products
        4. Rental income taxes
        5. Social housing rent
        6. Imputed rent
        7. Household rental income

        Returns:
            float: GDP value calculated using the income approach
        """
        total_sales = (self.firms.firm_data["Production"] * self.firms.firm_data["Price"]).sum()
        used_intermediate_inputs = self.firms.used_intermediate_inputs
        used_intermediate_inputs_costs = np.matmul(
            used_intermediate_inputs.T, self.firms.firm_data["Price"].values
        ).sum()

        wages = self.firms.firm_data["Total Wages Paid"].sum()

        taxes_on_production = self.firms.firm_data["Taxes paid on Production"].sum()

        operating_surplus = total_sales - wages - used_intermediate_inputs_costs - taxes_on_production

        # taxes_on_production_gov = self.central_government.central_gov_data["Taxes on Production"].values[0]

        taxes_on_products_gov = self.central_government.central_gov_data["Taxes on Products"].values[0]

        cg_rent_received = self.central_government.central_gov_data["Total Social Housing Rent"].values[0]

        cg_taxes_rental_income = self.central_government.central_gov_data["Rental Income Taxes"].values[0]

        rent_imputed = self.population.household_data["Rent Imputed"].sum()

        hh_rental_income = self.population.household_data["Rental Income from Real Estate"].sum()

        return (
            operating_surplus
            + wages
            + taxes_on_products_gov
            + cg_taxes_rental_income
            + cg_rent_received
            + rent_imputed
            + hh_rental_income
        )
