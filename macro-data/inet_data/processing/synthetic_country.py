from dataclasses import dataclass
from typing import Optional

import pandas as pd

from inet_data.configuration import CountryConfiguration
from inet_data.configuration.countries import Country
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
)
from inet_data.readers import DataReaders


@dataclass
class SyntheticCountry:
    """Container class for synthetic countries.

    Attributes:
        population: Synthetic population.
        firms: Synthetic firms.
        credit_market: Synthetic credit market.
        banks: Synthetic banks.
        central_bank: Synthetic central bank.
        central_government: Synthetic central government.
        government_entities: Synthetic government entities.
        housing_market: Synthetic housing market.
        dividend_payout_ratio: Dividend payout ratio.
        long_term_interest_rate: Long term interest rate.
        policy_rate_markup: Policy rate markup.
    """

    population: SyntheticPopulation
    firms: SyntheticFirms
    credit_market: SyntheticCreditMarket
    banks: SyntheticBanks
    central_bank: SyntheticCentralBank
    central_government: SyntheticCentralGovernment
    government_entities: SyntheticGovernmentEntities
    housing_market: SyntheticHousingMarket
    dividend_payout_ratio: float
    long_term_interest_rate: float
    policy_rate_markup: float

    @classmethod
    def eu_synthetic_country(
        cls,
        country: Country,
        year: int,
        country_configuration: CountryConfiguration,
        industries: list[str],
        readers: DataReaders,
        exogenous_country_data: Optional[dict[str, pd.DataFrame]],
        country_industry_data: dict[str, pd.DataFrame],
        year_range: int,
    ) -> "SyntheticCountry":
        """
        Create a synthetic country object for the European Union.

        Args:
            country: The country for which the synthetic country object is created.
            year: The year for which the synthetic country object is created.
            country_configuration: The configuration settings for the country.
            industries: The list of industries in the country.
            readers: The data readers used to read data for the synthetic country object.
            exogenous_country_data: The exogenous data for the country.
            country_industry_data: The industry data for the country.
            year_range: The range of years for which data is considered (determines the amount of data used to decide benefits setting).

        Returns:
            The synthetic country object.

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
        )

        central_bank = DefaultSyntheticCentralBank.from_readers(country, year, readers)

        population: SyntheticHFCSPopulation = SyntheticHFCSPopulation.from_readers(
            readers=readers,
            country_name=country,
            year=year,
            industry_data=country_industry_data,
            industries=industries,
            scale=country_configuration.scale,
            total_unemployment_benefits=total_unemployment_benefits,
            country_name_short=country.to_two_letter_code(),
        )

        firms = DefaultSyntheticFirms.from_readers(
            readers=readers,
            country_name=country,
            year=year,
            industry_data=country_industry_data,
            industries=industries,
            scale=country_configuration.scale,
            n_employees_per_industry=population.number_employees_by_industry,
            zero_initial_debt=country_configuration.firms_configuration.zero_initial_debt,
            zero_initial_deposits=country_configuration.firms_configuration.zero_initial_deposits,
        )

        banks = DefaultSyntheticBanks.from_readers(
            readers=readers,
            country_name=country,
            year=year,
            scale=country_configuration.scale,
            single_bank=country_configuration.single_bank,
        )

        match_individuals_with_firms_country(
            country=country,
            industries=industries,
            readers=readers,
            firms=firms,
            population=population,
            year=year,
        )

        match_firms_with_banks(firms=firms, banks=banks)

        match_households_with_banks(population=population, banks=banks)

        housing_data = set_housing_df(
            synthetic_population=population,
            rental_income_taxes=readers.oecd_econ.read_tau_income(country=country, year=year),
            social_housing_rent=population.social_housing_rent,
            total_imputed_rent=readers.icio[year].imputed_rents[country],
        )

        housing_market = DefaultSyntheticHousingMarket(
            year=year, housing_market_data=housing_data, country_name=country
        )

        # TODO : these functions do things that depend on the function parameters
        population.compute_household_wealth()

        population.compute_household_income(
            total_social_transfers=central_government.central_gov_data["Other Social Benefits"].values[0],
        )

        population.set_household_saving_rates()

        iot_consumption = country_industry_data["industry_vectors"]["Household Consumption in LCU"]

        vat = readers.world_bank.get_tau_vat(country, year)

        population.normalise_household_consumption(iot_hh_consumption=iot_consumption, vat=vat)

        weights_by_income = readers.oecd_econ.get_household_consumption_by_income_quantile(country=country, year=year)

        population.match_consumption_weights_by_income(
            weights_by_income=weights_by_income, iot_hh_consumption=iot_consumption, vat=vat
        )

        banks.initialise_deposits_and_loans(synthetic_population=population, synthetic_firms=firms)

        banks.initialise_rates_profits_liabilities(
            readers, **country_configuration.banks_configuration.interest_rates.dict()
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

        population.set_debt_installments(credit_market)

        firms.set_additional_initial_conditions(
            readers=readers,
            industry_data=country_industry_data,
            synthetic_banks=banks,
            synthetic_credit_market=credit_market,
        )

        central_government.update_fields(
            readers=readers,
            synthetic_banks=banks,
            synthetic_population=population,
            synthetic_firms=firms,
            industry_data=country_industry_data,
        )

        population.restrict()

        dividend_payout_ratio = readers.eurostat.dividend_payout_ratio(country=country, year=year)

        long_term_interest_rate = readers.oecd_econ.read_long_term_interest_rates(country=country, year=year)

        policy_rate_markup = readers.eurostat.firm_risk_premium(country=country, year=year)

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
        )
