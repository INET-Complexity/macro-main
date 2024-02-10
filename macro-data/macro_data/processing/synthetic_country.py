from dataclasses import dataclass
from typing import Optional

import pandas as pd

from macro_data.configuration import CountryDataConfiguration
from macro_data.configuration.countries import Country
from macro_data.processing.country_data import TaxData, ExogenousCountryData
from macro_data.processing.synthetic_banks.default_synthetic_banks import DefaultSyntheticBanks
from macro_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks
from macro_data.processing.synthetic_central_bank.default_synthetic_central_bank import DefaultSyntheticCentralBank
from macro_data.processing.synthetic_central_bank.synthetic_central_bank import SyntheticCentralBank
from macro_data.processing.synthetic_central_government.default_synthetic_central_government import (
    DefaultSyntheticCGovernment,
)
from macro_data.processing.synthetic_central_government.synthetic_central_government import SyntheticCentralGovernment
from macro_data.processing.synthetic_credit_market.synthetic_credit_market import SyntheticCreditMarket
from macro_data.processing.synthetic_firms.default_synthetic_firms import DefaultSyntheticFirms
from macro_data.processing.synthetic_firms.synthetic_firms import SyntheticFirms
from macro_data.processing.synthetic_government_entities.default_synthetic_government_entities import (
    DefaultSyntheticGovernmentEntities,
)
from macro_data.processing.synthetic_government_entities.synthetic_government_entities import SyntheticGovernmentEntities
from macro_data.processing.synthetic_housing_market.default_synthetic_housing_market import DefaultSyntheticHousingMarket
from macro_data.processing.synthetic_housing_market.synthetic_housing_market import SyntheticHousingMarket
from macro_data.processing.synthetic_matching.matching_firms_with_banks import match_firms_with_banks
from macro_data.processing.synthetic_matching.matching_households_with_banks import match_households_with_banks
from macro_data.processing.synthetic_matching.matching_households_with_houses import set_housing_df
from macro_data.processing.synthetic_matching.matching_individuals_with_firms import match_individuals_with_firms_country
from macro_data.processing.synthetic_population.hfcs_synthetic_population import SyntheticHFCSPopulation
from macro_data.processing.synthetic_population.synthetic_population import SyntheticPopulation
from macro_data.readers import DataReaders


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
        consumption_weights_by_income (pd.DataFrame): The consumption weights by income for the country.
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
    consumption_weights_by_income: pd.DataFrame

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
            firm_configuration=country_configuration.firms_configuration,
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
            housing_market,
        ) = cls.post_agents_init(
            banks=banks,
            central_government=central_government,
            country=country,
            country_configuration=country_configuration,
            country_industry_data=country_industry_data,
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
            consumption_weights_by_income=weights_by_income,
        )

    @classmethod
    def post_agents_init(
        cls,
        banks: SyntheticBanks,
        central_government: SyntheticCentralGovernment,
        country: Country,
        country_configuration: CountryDataConfiguration,
        country_industry_data: dict[str, pd.DataFrame],
        firms: SyntheticFirms,
        industries: list[str],
        population: SyntheticPopulation,
        tax_data: TaxData,
        total_rent: float,
        central_bank: SyntheticCentralBank,
        weights_by_income: pd.DataFrame,
    ) -> tuple[SyntheticCreditMarket, SyntheticHousingMarket]:
        """
        This function takes care of matching the different agents together and initialising the Credit and Housing markets.
        This function is separated because we may want to change the initialisation of firm parameters (in particular those which depend
        on function parameters), in which case we need to redo the matching and initialisation of the markets.

        Args:
            banks (SyntheticBanks): The synthetic banks.
            central_government (SyntheticCentralGovernment): The synthetic central government.
            country (Country): The country object representing the EU country.
            country_configuration (CountryDataConfiguration): The configuration data for the country.
            country_industry_data (dict[str, pd.DataFrame]): The industry data for the country.
            firms (SyntheticFirms): The synthetic firms.
            industries (list[str]): The list of industries in the country.
            population (SyntheticPopulation): The synthetic population.
            tax_data (TaxData): The tax data for the country.
            total_rent (float): The total rent in the country.
            central_bank (SyntheticCentralBank): The synthetic central bank.
            weights_by_income (pd.DataFrame): The weights by income for the country.

        Returns:
            tuple[SyntheticCreditMarket, SyntheticHousingMarket]: A tuple containing the synthetic credit market,
            exogenous data, and synthetic housing market.
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
        independents = None
        # here this only changes if we change the independents of the function (e.g. income, debt)
        # not worth it to change it now

        policy_rate = central_bank.central_bank_data["Policy Rate"].values[0]

        cls.initialise_pop_wealth_income(
            banks=banks,
            central_government=central_government,
            country_industry_data=country_industry_data,
            firms=firms,
            population=population,
            vat=tax_data.value_added_tax,
            weights_by_income=weights_by_income,
            independents=independents,
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
        return (
            credit_market,
            housing_market,
        )

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
        Initializes the credit market.

        Args:
            cls: The class object.
            banks: The synthetic banks object.
            central_government: The synthetic central government object.
            country_configuration: The country data configuration object.
            country_industry_data: The dictionary containing country industry data.
            firms: The synthetic firms object.
            population: The synthetic population object.
            tax_data: The tax data object.
            risk_premium: The risk premium for interest rates.
            policy_rate: The policy rate for interest rates.

        Returns:
            The initialized synthetic credit market object.
        """

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
        population.set_debt_installments(credit_market.credit_market_data)
        firms.set_additional_initial_conditions(
            tax_data=tax_data,
            industry_data=country_industry_data,
            synthetic_banks=banks,
            credit_market_data=credit_market.credit_market_data,
        )
        central_government.update_fields(
            synthetic_banks=banks,
            synthetic_population=population,
            synthetic_firms=firms,
            industry_data=country_industry_data,
            tax_data=tax_data,
        )
        return credit_market

    @classmethod
    def initialise_pop_wealth_income(
        cls,
        banks: SyntheticBanks,
        central_government: SyntheticCentralGovernment,
        country_industry_data: dict[str, pd.DataFrame],
        firms: SyntheticFirms,
        population: SyntheticPopulation,
        vat: float,
        weights_by_income: pd.DataFrame,
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
        banks.initialise_deposits_and_loans(
            synthetic_population=population,
            firm_deposits=firms.firm_data["Deposits"].values,
            firm_debt=firms.firm_data["Debt"].values,
        )
        # bank tax rate set to same as corporate tax rate

    def reset_firm_function_dependent(
        self,
        capital_inputs_utilisation_rate: float,
        initial_inventory_to_input_fraction: float,
        intermediate_inputs_utilisation_rate: float,
        zero_initial_debt: bool,
        zero_initial_deposits: bool,
    ):
        """
        Resets the function parameters of the firms and initializes the credit market, exogenous data, and housing market.
        These must be reinitialised because changing the firm function parameters will change their balance sheet, which in turn
        will impact household finances, and thus the credit market, exogenous data, and housing market.

        Args:
            capital_inputs_utilisation_rate (float): The rate of capital inputs utilization.
            initial_inventory_to_input_fraction (float): The fraction of initial inventory to input.
            intermediate_inputs_utilisation_rate (float): The rate of intermediate inputs utilization.
            zero_initial_debt (bool): Flag indicating whether to set initial debt to zero.
            zero_initial_deposits (bool): Flag indicating whether to set initial deposits to zero.
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

        credit_market, housing_market = self.post_agents_init(
            banks=self.banks,
            central_government=self.central_government,
            country=self.country_name,
            country_configuration=self.country_configuration,
            country_industry_data=self.industry_data,
            firms=self.firms,
            industries=self.industries,
            population=self.population,
            tax_data=self.tax_data,
            total_rent=total_rent,
            central_bank=self.central_bank,
            weights_by_income=self.consumption_weights_by_income,
        )

        self.credit_market = credit_market
        self.housing_market = housing_market
