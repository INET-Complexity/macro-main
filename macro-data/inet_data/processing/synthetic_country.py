from dataclasses import dataclass
from typing import Optional

import pandas as pd

from inet_data.configuration import CountryDataConfiguration
from inet_data.configuration.countries import Country
from inet_data.processing.country_data import TaxData, ExogenousCountryData
from inet_data.processing.synthetic_banks.default_synthetic_banks import DefaultSyntheticBanks
from inet_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks
from inet_data.processing.synthetic_central_bank.default_synthetic_central_bank import DefaultSyntheticCentralBank
from inet_data.processing.synthetic_central_bank.synthetic_central_bank import SyntheticCentralBank
from inet_data.processing.synthetic_central_government.default_synthetic_central_government import (
    DefaultSyntheticCGovernment,
)
from inet_data.processing.synthetic_central_government.synthetic_central_government import SyntheticCentralGovernment
from inet_data.processing.synthetic_credit_market.synthetic_credit_market import SyntheticCreditMarket
from inet_data.processing.synthetic_firms.default_synthetic_firms import DefaultSyntheticFirms
from inet_data.processing.synthetic_firms.synthetic_firms import SyntheticFirms
from inet_data.processing.synthetic_government_entities.default_synthetic_government_entities import (
    DefaultSyntheticGovernmentEntities,
)
from inet_data.processing.synthetic_government_entities.synthetic_government_entities import SyntheticGovernmentEntities
from inet_data.processing.synthetic_housing_market.default_synthetic_housing_market import DefaultSyntheticHousingMarket
from inet_data.processing.synthetic_housing_market.synthetic_housing_market import SyntheticHousingMarket
from inet_data.processing.synthetic_matching.matching_firms_with_banks import match_firms_with_banks
from inet_data.processing.synthetic_matching.matching_households_with_banks import match_households_with_banks
from inet_data.processing.synthetic_matching.matching_households_with_houses import set_housing_df
from inet_data.processing.synthetic_matching.matching_individuals_with_firms import match_individuals_with_firms_country
from inet_data.processing.synthetic_population.hfcs_synthetic_population import SyntheticHFCSPopulation
from inet_data.processing.synthetic_population.synthetic_population import SyntheticPopulation
from inet_data.readers import DataReaders


@dataclass
class SyntheticCountry:
    """Container class for synthetic countries.

    Attributes:
        population (SyntheticPopulation): Synthetic population.
        firms (SyntheticFirms): Synthetic firms.
        credit_market (SyntheticCreditMarket): Synthetic credit market.
        banks (SyntheticBanks): Synthetic banks.
        central_bank (SyntheticCentralBank): Synthetic central bank.
        central_government (SyntheticCentralGovernment): Synthetic central government.
        government_entities (SyntheticGovernmentEntities): Synthetic government entities.
        housing_market (SyntheticHousingMarket): Synthetic housing market.
        dividend_payout_ratio (float): Dividend payout ratio.
        long_term_interest_rate (float): Long term interest rate.
        policy_rate_markup (float): Policy rate markup.
        industry_data (dict[str, pd.DataFrame]): Industry data for the country (includes various industry data).
        goods_criticality_matrix (pd.DataFrame): The goods criticality matrix.
        tax_data (TaxData): Tax data.
        exogenous_data (ExogenousCountryData): Exogenous data.
        scale (int): The scale of the synthetic country.
        country_name (Country): The name of the country.
        country_configuration (CountryDataConfiguration): The configuration settings for the country.
        industries (list[str]): The list of industries in the country.
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
    industry_data: dict[str, pd.DataFrame]
    goods_criticality_matrix: pd.DataFrame
    tax_data: TaxData
    exogenous_data: ExogenousCountryData
    scale: int
    country_name: Country
    country_configuration: CountryDataConfiguration
    industries: list[str]
    weights_by_income: pd.DataFrame

    @classmethod
    def eu_synthetic_country(
        cls,
        country: Country,
        year: int,
        country_configuration: CountryDataConfiguration,
        industries: list[str],
        readers: DataReaders,
        exogenous_country_data: dict[str, pd.DataFrame],
        country_industry_data: dict[str, pd.DataFrame],
        year_range: int,
        goods_criticality_matrix: pd.DataFrame,
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
            goods_criticality_matrix: The goods criticality matrix.

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

        exogenous_data = ExogenousCountryData(**exogenous_country_data)

        tax_data = TaxData.from_readers(readers, country, year)

        total_imputed_rent = readers.icio[year].imputed_rents[country]

        dividend_payout_ratio = readers.eurostat.dividend_payout_ratio(country=country, year=year)
        long_term_interest_rate = readers.oecd_econ.read_long_term_interest_rates(country=country, year=year)
        policy_rate_markup = readers.eurostat.firm_risk_premium(country=country, year=year)

        weights_by_income = readers.oecd_econ.get_household_consumption_by_income_quantile(country=country, year=year)

        (
            credit_market,
            exogenous_data,
            housing_market,
        ) = cls.eu_post_agents_init(
            banks=banks,
            central_government=central_government,
            country=country,
            country_configuration=country_configuration,
            country_industry_data=country_industry_data,
            exogenous_data=exogenous_data,
            firms=firms,
            industries=industries,
            population=population,
            tax_data=tax_data,
            total_rent=total_imputed_rent,
            central_bank=central_bank,
            weights_by_income=weights_by_income,
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
            exogenous_data=exogenous_data,
            scale=country_configuration.scale,
            country_name=country,
            country_configuration=country_configuration,
            industries=industries,
            weights_by_income=weights_by_income,
        )

    @classmethod
    def eu_post_agents_init(
        cls,
        banks: SyntheticBanks,
        central_government: SyntheticCentralGovernment,
        country: Country,
        country_configuration: CountryDataConfiguration,
        country_industry_data: dict[str, pd.DataFrame],
        exogenous_data: ExogenousCountryData,
        firms: SyntheticFirms,
        industries: list[str],
        population: SyntheticPopulation,
        tax_data: TaxData,
        total_rent: float,
        central_bank: SyntheticCentralBank,
        weights_by_income: pd.DataFrame,
    ):
        income_taxes = tax_data.income_tax
        employee_social_contribution_taxes = tax_data.employee_social_insurance_tax
        match_individuals_with_firms_country(
            industries=industries,
            income_taxes=income_taxes,
            employee_social_contribution_taxes=employee_social_contribution_taxes,
            firms=firms,
            population=population,
        )
        match_firms_with_banks(firms=firms, banks=banks)
        match_households_with_banks(population=population, banks=banks)
        housing_data = set_housing_df(
            synthetic_population=population,
            rental_income_taxes=tax_data.income_tax,
            social_housing_rent=population.social_housing_rent,
            total_imputed_rent=total_rent,
        )
        housing_market = DefaultSyntheticHousingMarket(housing_market_data=housing_data, country_name=country)
        population.compute_household_wealth()
        # TODO : these functions do things that depend on the function parameters
        independents = None
        # here this only changes if we change the independents of the function (e.g. income, debt)
        # not worth it to change it now

        policy_rate = central_bank.central_bank_data["Policy Rate"].values[0]

        credit_market = cls.eu_population_wealth_post_init(
            banks=banks,
            central_government=central_government,
            country_configuration=country_configuration,
            country_industry_data=country_industry_data,
            firms=firms,
            population=population,
            vat=tax_data.value_added_tax,
            independents=independents,
            tax_data=tax_data,
            risk_premium=tax_data.risk_premium,
            policy_rate=policy_rate,
            weights_by_income=weights_by_income,
        )
        return (
            credit_market,
            exogenous_data,
            housing_market,
        )

    @classmethod
    def eu_population_wealth_post_init(
        cls,
        banks: SyntheticBanks,
        central_government: SyntheticCentralGovernment,
        country_configuration: CountryDataConfiguration,
        country_industry_data: dict[str, pd.DataFrame],
        firms: SyntheticFirms,
        population: SyntheticPopulation,
        tax_data: TaxData,
        weights_by_income: pd.DataFrame,
        vat: float,
        risk_premium: float,
        policy_rate: float,
        independents: Optional[list[str]] = None,
    ):
        population.compute_household_wealth(independents=independents)
        population.compute_household_income(
            total_social_transfers=central_government.central_gov_data["Other Social Benefits"].values[0],
            independents=independents,
        )
        population.set_household_saving_rates(independents=independents)
        iot_consumption = country_industry_data["industry_vectors"]["Household Consumption in LCU"]
        population.normalise_household_consumption(
            iot_hh_consumption=iot_consumption, vat=vat, independents=independents
        )
        population.match_consumption_weights_by_income(
            weights_by_income=weights_by_income, iot_hh_consumption=iot_consumption, vat=vat
        )
        banks.initialise_deposits_and_loans(synthetic_population=population, synthetic_firms=firms)

        # bank tax rate set to same as corporate tax rate

        tau_bank = tax_data.profit_tax

        banks.initialise_rates_profits_liabilities(
            policy_rate=policy_rate,
            tau_bank=tau_bank,
            risk_premium=risk_premium,
            **country_configuration.banks_configuration.interest_rates.dict()
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
            tax_data=tax_data,
            industry_data=country_industry_data,
            synthetic_banks=banks,
            synthetic_credit_market=credit_market,
        )
        central_government.update_fields(
            synthetic_banks=banks,
            synthetic_population=population,
            synthetic_firms=firms,
            industry_data=country_industry_data,
            tax_data=tax_data,
        )
        return credit_market

    def reset_firm_function_dependent(
        self,
        capital_inputs_utilisation_rate: float,
        iniital_inventory_to_inputs_fraction: float,
        intermediate_inputs_utilisation_rate: float,
        zero_initial_debt: bool,
        zero_initial_deposits: bool,
    ):
        self.firms.reset_function_parameters(
            capital_inputs_utilisation_rate=capital_inputs_utilisation_rate,
            initial_inventory_to_input_fraction=iniital_inventory_to_inputs_fraction,
            intermediate_inputs_utilisation_rate=intermediate_inputs_utilisation_rate,
            zero_initial_debt=zero_initial_debt,
            zero_initial_deposits=zero_initial_deposits,
        )

        housing_data = self.housing_market.housing_market_data
        owned_houses = housing_data["Is Owner-Occupied"]
        total_rent = housing_data.loc[owned_houses, "Rent"].sum()

        self.eu_post_agents_init(
            banks=self.banks,
            central_government=self.central_government,
            country=self.country_name,
            country_configuration=self.country_configuration,
            country_industry_data=self.industry_data,
            exogenous_data=self.exogenous_data,
            firms=self.firms,
            industries=self.industries,
            population=self.population,
            central_bank=self.central_bank,
            tax_data=self.tax_data,
            weights_by_income=self.weights_by_income,
            total_rent=total_rent,
        )
