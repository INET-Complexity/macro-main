from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from macro_data.configuration import CountryDataConfiguration
from macro_data.configuration.countries import Country
from macro_data.processing import set_housing_df
from macro_data.processing.country_data import TaxData
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
from macro_data.processing.synthetic_goods_market.synthetic_goods_market import SyntheticGoodsMarket
from macro_data.processing.synthetic_government_entities.default_synthetic_government_entities import (
    DefaultSyntheticGovernmentEntities,
)
from macro_data.processing.synthetic_government_entities.synthetic_government_entities import (
    SyntheticGovernmentEntities,
)
from macro_data.processing.synthetic_housing_market.default_synthetic_housing_market import (
    DefaultSyntheticHousingMarket,
)
from macro_data.processing.synthetic_housing_market.synthetic_housing_market import SyntheticHousingMarket
from macro_data.processing.synthetic_matching.matching_firms_with_banks import (
    match_firms_with_banks_optimal,
)
from macro_data.processing.synthetic_matching.matching_households_with_banks import (
    match_households_with_banks_optimal,
)
from macro_data.processing.synthetic_matching.matching_individuals_with_firms import (
    match_individuals_with_firms_country,
)
from macro_data.processing.synthetic_population.hfcs_synthetic_population import SyntheticHFCSPopulation
from macro_data.processing.synthetic_population.synthetic_population import SyntheticPopulation
from macro_data.readers import DataReaders
from macro_data.readers.exogenous_data import ExogenousCountryData


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
        synthetic_goods_market (SyntheticGoodsMarket): Synthetic goods market.
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
    ) -> "SyntheticCountry":
        """
        Create a synthetic country object for the European Union.

        Args:
            country: The country for which the synthetic country object is created.
            year: The year for which the synthetic country object is created.
            quarter: The quarter for which the synthetic country object is created.
            country_configuration: The configuration settings for the country.
            industries: The list of industries in the country.
            readers: The data readers used to read data for the synthetic country object.
            exogenous_country_data: The exogenous data for the country.
            country_industry_data: The industry data for the country.
            year_range: The range of years for which data is considered (determines the amount of data used to
                        decide benefits setting).
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
            quarter=quarter,
            exogenous_country_data=exogenous_country_data,
            industry_data=country_industry_data,
            single_government_entity=country_configuration.single_government_entity,
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
        )

        synthetic_goods_market = SyntheticGoodsMarket.from_readers(
            country_name=country, year=year, quarter=quarter, readers=readers, exogenous_data=exogenous_country_data
        )

        tax_data = TaxData.from_readers(readers, country, year)

        total_imputed_rent = readers.icio[year].imputed_rents[country]

        dividend_payout_ratio = readers.eurostat.dividend_payout_ratio(country=country, year=year)
        long_term_interest_rate = readers.oecd_econ.read_long_term_interest_rates(country=country, year=year)
        policy_rate_markup = readers.eurostat.firm_risk_premium(country=country, year=year)

        weights_by_income = readers.oecd_econ.get_household_consumption_by_income_quantile(country=country, year=year)
        cls.match_households_firms_banks(banks, firms, industries, population, tax_data)

        housing_data = set_housing_df(
            synthetic_population=population,
            rental_income_taxes=tax_data.income_tax,
            social_housing_rent=population.social_housing_rent,
            total_imputed_rent=total_imputed_rent,
        )

        housing_market = DefaultSyntheticHousingMarket(country, housing_data)

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
        )

    @classmethod
    def proxied_synthetic_country(
        cls,
        country: Country,
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
    ) -> "SyntheticCountry":
        """
        Create a synthetic country object for a country using a European Union country as a proxy for population.

        Args:
            country: The country for which the synthetic country object is created.
            proxy_country: The proxy country to use for the synthetic country object.
            year: The year for which the synthetic country object is created.
            quarter: The quarter for which the synthetic country object is created.
            country_configuration: The configuration settings for the country.
            industries: The list of industries in the country.
            readers: The data readers used to read data for the synthetic country object.
            exogenous_country_data: The exogenous data for the country.
            country_industry_data: The industry data for the country.
            year_range: The range of years for which data is considered
             (determines the amount of data used to decide benefits setting).
            goods_criticality_matrix: The goods criticality matrix.
            proxy_inflation_data: The inflation data for the proxy country.

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
            quarter=quarter,
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

        weights_by_income = readers.oecd_econ.get_household_consumption_by_income_quantile(country=country, year=year)

        cls.match_households_firms_banks(banks, firms, industries, population, tax_data)

        housing_data = set_housing_df(
            synthetic_population=population,
            rental_income_taxes=tax_data.income_tax,
            social_housing_rent=population.social_housing_rent,
            total_imputed_rent=total_imputed_rent,
        )

        housing_market = DefaultSyntheticHousingMarket(country, housing_data)

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
        return self.firms.number_of_firms_by_industry

    @property
    def n_buyers(self):
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
    ):
        population.set_wealth_distribution_function(independents=independents)

        population.compute_household_income(
            total_social_transfers=central_government.central_gov_data["Other Social Benefits"].values[0],
            independents=independents,
        )
        population.set_household_saving_rates(independents=independents)

        household_investment = country_industry_data["industry_vectors"]["Household Capital Inputs in LCU"].values

        population.set_household_investment_rates(
            household_investment=household_investment, capital_formation_taxrate=tax_data.capital_formation_tax
        )
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
            capital_inputs_utilisation_rate (float): The rate of capital inputs utilisation.
            initial_inventory_to_input_fraction (float): The fraction of initial inventory to input.
            intermediate_inputs_utilisation_rate (float): The rate of intermediate inputs utilisation.
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
