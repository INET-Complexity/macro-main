"""Module implementing the country-level economic model.

This module provides the core Country class that integrates all economic agents,
markets, and processes within a single national economy. It orchestrates:

1. Economic Agents:
   - Individuals (workers, consumers)
   - Households (consumption, investment, housing decisions)
   - Firms (production, pricing, employment)
   - Banks (lending, deposit-taking)
   - Government entities (public spending, services)
   - Central bank (monetary policy)

2. Markets:
   - Labor market (employment, wages)
   - Credit market (loans, interest rates)
   - Housing market (property sales, rentals)
   - Goods market (production, consumption)

3. Economic Processes:
   - Production and consumption
   - Price formation and inflation
   - Wage determination
   - Credit creation and allocation
   - Fiscal and monetary policy
   - International trade

The model runs in discrete timesteps (typically quarterly but configurable),
with each iteration updating all agents and markets in sequence according
to their behavioral rules and interactions.
"""

import logging
from copy import deepcopy
from typing import Optional

import h5py
import numpy as np
import pandas as pd

from macro_data import SyntheticCountry
from macromodel.agents.agent import Agent
from macromodel.agents.banks.banks import Banks
from macromodel.agents.central_bank.central_bank import CentralBank
from macromodel.agents.central_government.central_government import CentralGovernment
from macromodel.agents.firms import Firms
from macromodel.agents.government_entities.government_entities import GovernmentEntities
from macromodel.agents.households.households import Households
from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.agents.individuals.individuals import Individuals
from macromodel.configurations import CountryConfiguration
from macromodel.economy.economy import Economy
from macromodel.exchange_rates import ExchangeRates
from macromodel.exogenous.exogenous import Exogenous
from macromodel.markets.credit_market.credit_market import CreditMarket
from macromodel.markets.housing_market.housing_market import HousingMarket
from macromodel.markets.labour_market.labour_market import LabourMarket
from macromodel.rest_of_the_world import RestOfTheWorld
from macromodel.util.get_histogram import get_histogram


class Country:
    """A complete national economy with interacting agents and markets.

    This class represents a single country's economy, integrating all economic agents
    (individuals, households, firms, banks, government) and markets (labor, credit,
    housing, goods). It manages their interactions and evolution over time.

    The economy operates in discrete timesteps, with each iteration:
    1. Updating exchange rates and monetary conditions
    2. Running agent-level planning and decision-making
    3. Clearing all markets sequentially
    4. Recording economic outcomes and updating metrics

    The timestep length is configurable (typically quarterly) and determined by
    how the input data is preprocessed (see ICIOReader's yearly_factor parameter).

    Attributes:
        country_name (str): Identifier for the country
        scale (int): Scaling factor for population-based calculations
        individuals (Individuals): Population of individual economic agents
        households (Households): Collection of household units
        firms (Firms): Collection of producing firms
        central_government (CentralGovernment): Fiscal authority
        government_entities (GovernmentEntities): Public sector bodies
        banks (Banks): Commercial banking sector
        central_bank (CentralBank): Monetary authority
        economy (Economy): Aggregate economic metrics and tracking
        labour_market (LabourMarket): Employment matching and wage setting
        credit_market (CreditMarket): Lending and borrowing
        housing_market (HousingMarket): Property sales and rentals
        exchange_rate_usd_to_lcu (float): Exchange rate to USD
        exogenous (Exogenous): External economic conditions
        forecasting_window (int): Periods ahead for forecasting
        assume_zero_growth (bool): Whether to assume no growth
        assume_zero_noise (bool): Whether to assume no random shocks
        configuration (CountryConfiguration): Model parameters
    """

    country_name: str
    scale: int
    individuals: Individuals
    households: Households
    firms: Firms
    central_government: CentralGovernment
    government_entities: GovernmentEntities
    banks: Banks
    central_bank: CentralBank
    economy: Economy
    labour_market: LabourMarket
    credit_market: CreditMarket
    housing_market: HousingMarket
    exchange_rate_usd_to_lcu: float
    exogenous: Exogenous
    forecasting_window: int
    assume_zero_growth: bool
    assume_zero_noise: bool
    configuration: CountryConfiguration

    def __init__(
        self,
        country_name: str,
        scale: int,
        individuals: Individuals,
        households: Households,
        firms: Firms,
        central_government: CentralGovernment,
        government_entities: GovernmentEntities,
        banks: Banks,
        central_bank: CentralBank,
        economy: Economy,
        labour_market: LabourMarket,
        credit_market: CreditMarket,
        housing_market: HousingMarket,
        exogenous: Exogenous,
        running_multiple_countries: bool,
        configuration: CountryConfiguration = CountryConfiguration(),
        forecasting_window: int = 60,
        assume_zero_growth: bool = False,
        assume_zero_noise: bool = False,
        add_emissions: bool = False,
        emission_factors_lcu: Optional[np.ndarray] = None,
        emitting_indices: Optional[np.ndarray] = None,
    ):
        """Initialize a new country economy.

        Args:
            country_name (str): Country identifier
            scale (int): Population scaling factor
            individuals (Individuals): Individual agents
            households (Households): Household units
            firms (Firms): Producing firms
            central_government (CentralGovernment): Fiscal authority
            government_entities (GovernmentEntities): Public sector
            banks (Banks): Banking sector
            central_bank (CentralBank): Monetary authority
            economy (Economy): Economic tracking
            labour_market (LabourMarket): Employment market
            credit_market (CreditMarket): Lending market
            housing_market (HousingMarket): Property market
            exogenous (Exogenous): External conditions
            running_multiple_countries (bool): If True, part of multi-country sim
            configuration (CountryConfiguration): Model parameters
            forecasting_window (int): Forecast periods
            assume_zero_growth (bool): If True, assume no growth
            assume_zero_noise (bool): If True, assume no shocks
            add_emissions (bool): If True, track emissions
            emission_factors_lcu (Optional[np.ndarray]): Emission factors
            emitting_indices (Optional[np.ndarray]): Industry indices that emit
        """
        # Parameters
        self.country_name = country_name
        self.scale = scale

        # Agents
        self.individuals = individuals
        self.households = households
        self.firms = firms
        self.central_government = central_government
        self.government_entities = government_entities
        self.banks = banks
        self.central_bank = central_bank

        # The economy
        self.economy = economy

        # Markets
        self.labour_market = labour_market
        self.credit_market = credit_market
        self.housing_market = housing_market

        # Exchange rate
        self.exchange_rate_usd_to_lcu = 1

        # Exogenous data
        self.exogenous = exogenous

        # Configuration
        self.forecasting_window = forecasting_window
        self.assume_zero_growth = assume_zero_growth
        self.assume_zero_noise = assume_zero_noise

        self.running_multiple_countries = running_multiple_countries

        self.configuration = configuration

        self.add_emissions = add_emissions
        self.emission_factors_lcu = emission_factors_lcu
        self.emitting_indices = emitting_indices

    @classmethod
    def from_pickled_country(
        cls,
        synthetic_country: SyntheticCountry,
        country_configuration: CountryConfiguration,
        exchange_rates: ExchangeRates,
        country_name: str,
        all_country_names: list[str],
        industries: list[str],
        initial_year: int,
        t_max: int,
        running_multiple_countries: bool,
        emission_factors_usd: np.ndarray,
    ) -> "Country":
        """Create a Country instance from preprocessed synthetic data.

        This factory method constructs a complete country economy from preprocessed
        data, initializing all agents and markets with calibrated starting conditions.

        Args:
            synthetic_country (SyntheticCountry): Preprocessed country data
            country_configuration (CountryConfiguration): Model parameters
            exchange_rates (ExchangeRates): Exchange rate dynamics
            country_name (str): Country identifier
            all_country_names (list[str]): All countries in simulation
            industries (list[str]): Industry sectors
            initial_year (int): Starting year
            t_max (int): Maximum simulation periods
            running_multiple_countries (bool): If True, part of multi-country sim
            emission_factors_usd (np.ndarray): Emission factors in USD

        Returns:
            Country: Initialized country economy
        """
        scale = synthetic_country.scale

        emission_industries = ["B05a", "B05b", "B05c", "C19"]
        add_emissions = all([industry in industries for industry in emission_industries])

        if add_emissions:
            emitting_indices = np.array([list(industries).index(industry) for industry in emission_industries])
            emission_factors_lcu = synthetic_country.emission_factors.emissions_array
        else:
            emitting_indices = None
            emission_factors_lcu = None

        n_industries = len(industries)

        synthetic_population = synthetic_country.population
        individuals = Individuals.from_pickled_agent(
            synthetic_population=synthetic_population,
            configuration=country_configuration.individuals,
            country_name=country_name,
            all_country_names=all_country_names,
            n_industries=n_industries,
            scale=scale,
        )

        initial_consumption_by_industry = synthetic_country.industry_data["industry_vectors"][
            "Household Consumption in LCU"
        ]
        households = Households.from_pickled_agent(
            synthetic_population=synthetic_population,
            synthetic_country=synthetic_country,
            configuration=country_configuration.households,
            country_name=country_name,
            all_country_names=all_country_names,
            industries=industries,
            initial_consumption_by_industry=initial_consumption_by_industry,
            value_added_tax=synthetic_country.tax_data.value_added_tax,
            scale=scale,
            add_emissions=add_emissions,
        )

        average_initial_price = synthetic_country.industry_data["industry_vectors"]["Average Initial Price"].values
        firms = Firms.from_pickled_agent(
            synthetic_firms=synthetic_country.firms,
            configuration=country_configuration.firms,
            country_name=country_name,
            all_country_names=all_country_names,
            goods_criticality_matrix=synthetic_country.goods_criticality_matrix,
            average_initial_price=average_initial_price,
            industries=industries,
            add_emissions=add_emissions,
        )

        taxes_less_subsidies = synthetic_country.industry_data["industry_vectors"]["Taxes Less Subsidies Rates"].values

        n_unemployed = (individuals.states["Activity Status"] == ActivityStatus.UNEMPLOYED).sum()

        central_government = CentralGovernment.from_pickled_agent(
            synthetic_central_government=synthetic_country.central_government,
            configuration=country_configuration.central_government,
            country_name=country_name,
            all_country_names=all_country_names,
            taxes_net_subsidies=taxes_less_subsidies,
            number_of_unemployed_individuals=n_unemployed,
            tax_data=synthetic_country.tax_data,
            n_industries=n_industries,
        )

        government_entities = GovernmentEntities.from_pickled_agent(
            synthetic_government_entities=synthetic_country.government_entities,
            configuration=country_configuration.government_entities,
            country_name=country_name,
            all_country_names=all_country_names,
            n_industries=n_industries,
            add_emissions=add_emissions,
            emission_factors_lcu=emission_factors_lcu,
            emitting_indices=emitting_indices,
        )

        banks = Banks.from_pickled_agent(
            synthetic_banks=synthetic_country.banks,
            configuration=country_configuration.banks,
            policy_rate_markup=synthetic_country.policy_rate_markup,
            country_name=country_name,
            all_country_names=all_country_names,
            n_industries=n_industries,
            scale=scale,
        )

        central_bank = CentralBank.from_pickled_agent(
            synthetic_central_bank=synthetic_country.central_bank,
            configuration=country_configuration.central_bank,
            country_name=country_name,
            all_country_names=all_country_names,
            n_industries=n_industries,
        )

        exogenous = Exogenous.from_pickled_agent(
            synthetic_country=synthetic_country,
            exchange_rates=exchange_rates,
            country_name=country_name,
            initial_year=initial_year,
            t_max=t_max,
        )

        economy = Economy.from_agents(
            country_name=country_name,
            all_country_names=all_country_names,
            economy_configuration=country_configuration.economy,
            individuals=individuals,
            firms=firms,
            central_government=central_government,
            government_entities=government_entities,
            households=households,
            industry_vectors=synthetic_country.industry_data["industry_vectors"],
            exogenous=exogenous,
        )

        labour_market = LabourMarket.from_agents(
            individuals=individuals,
            labour_market_configuration=country_configuration.labour_market,
            country_name=country_name,
            n_industries=n_industries,
        )

        credit_market = CreditMarket.from_pickled_market(
            synthetic_credit_market=synthetic_country.credit_market,
            credit_market_configuration=country_configuration.credit_market,
            country_name=country_name,
        )

        housing_market = HousingMarket.from_pickled_market(
            synthetic_housing_market=synthetic_country.housing_market,
            housing_market_configuration=country_configuration.housing_market,
            country_name=country_name,
            scale=scale,
        )

        return cls(
            country_name=country_name,
            scale=scale,
            individuals=individuals,
            households=households,
            firms=firms,
            central_government=central_government,
            government_entities=government_entities,
            banks=banks,
            central_bank=central_bank,
            economy=economy,
            labour_market=labour_market,
            credit_market=credit_market,
            housing_market=housing_market,
            exogenous=exogenous,
            forecasting_window=country_configuration.forecasting_window,
            assume_zero_growth=country_configuration.assume_zero_growth,
            assume_zero_noise=country_configuration.assume_zero_noise,
            running_multiple_countries=running_multiple_countries,
            configuration=country_configuration,
            add_emissions=add_emissions,
            emission_factors_lcu=emission_factors_lcu,
            emitting_indices=emitting_indices,
        )

    def reset(self, configuration: CountryConfiguration) -> None:
        """Reset the country economy to its initial state.

        Resets all agents, markets, and state variables to their initial conditions,
        optionally with new configuration parameters.

        Args:
            configuration (CountryConfiguration): New model parameters
        """
        self.forecasting_window = configuration.forecasting_window
        self.assume_zero_growth = configuration.assume_zero_growth
        self.assume_zero_noise = configuration.assume_zero_noise

        self.individuals.reset(configuration=configuration.individuals)
        self.households.reset(configuration=configuration.households)
        self.firms.reset(configuration=configuration.firms)
        self.central_government.reset(configuration=configuration.central_government)
        self.government_entities.reset(configuration=configuration.government_entities)
        self.banks.reset(configuration=configuration.banks)
        self.central_bank.reset(configuration=configuration.central_bank)

        self.economy.reset(configuration=configuration.economy)

        self.labour_market.reset(configuration=configuration.labour_market)
        self.credit_market.reset(configuration=configuration.credit_market)
        self.housing_market.reset(configuration=configuration.housing_market)

        self.exogenous.reset()

        self.configuration = deepcopy(configuration)

    def initialisation_phase(self, exchange_rate_usd_to_lcu: float) -> None:
        """Initialize the monthly economic cycle.

        Sets up the initial conditions for the current timestep, including
        exchange rates and firm counts.

        Args:
            exchange_rate_usd_to_lcu (float): Current exchange rate to USD
        """
        self.exchange_rate_usd_to_lcu = exchange_rate_usd_to_lcu
        self.firms.update_number_of_firms()

    def estimation_phase(self) -> None:
        """Run economic forecasting and estimation.

        Updates agent expectations about growth, inflation, and other key
        economic variables used in decision-making.
        """
        self.economy.set_estimates(
            exogenous_growth=self.exogenous.national_accounts_before["Real Gross Output (Growth)"].values,
            exogenous_inflation=self.exogenous.inflation_before,
            exogenous_hpi_growth=self.exogenous.house_price_index_before,
            exogenous_ppi_inflation_during=self.exogenous.national_accounts_during["PPI (Growth)"].values.flatten(),
            exogenous_cpi_inflation_during=self.exogenous.national_accounts_during["CPI (Growth)"].values.flatten(),
            exogenous_growth_during=self.exogenous.national_accounts_during[
                "Real Gross Output (Growth)"
            ].values.flatten(),
            forecasting_window=self.forecasting_window,
            assume_zero_growth=self.assume_zero_growth,
            assume_zero_noise=self.assume_zero_noise,
        )
        self.firms.set_estimates(
            previous_average_good_prices=self.economy.ts.current("good_prices"),
            current_estimated_growth=self.economy.ts.current("estimated_growth")[0],
        )

    def target_setting_phase(self) -> None:
        """Set agent-level targets and plans.

        Updates production targets, wage offers, and reservation wages based
        on current conditions and expectations.
        """
        # Firms set production targets
        self.firms.set_targets(
            bank_overdraft_rate_on_firm_deposits=self.banks.ts.current("overdraft_rate_on_firm_deposits"),
            estimated_growth=self.economy.ts.current("estimated_growth")[0],
            estimated_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
            current_good_prices=self.economy.ts.current("good_prices"),
        )

        # Changes in labour productivity
        self.firms.ts.wage_tightness_markup.append(self.firms.compute_wages_markup())

        # Firms determine the wages they're willing to pay new employees
        """
        self.firms.states["offered_wage_function"] = self.firms.compute_offered_wage_function(
            corresponding_firm=self.individuals.states["Corresponding Firm ID"],
            current_individual_labour_inputs=self.individuals.ts.current("labour_inputs"),
            previous_employee_income=self.individuals.ts.current("employee_income"),
            unemployment_benefits_by_individual=self.central_government.ts.current(
                "unemployment_benefits_by_individual"
            )[0],
            income_taxes=self.central_government.states["Income Tax"],
            employee_social_insurance_tax=self.central_government.states["Employee Social Insurance Tax"],
            employer_social_insurance_tax=self.central_government.states["Employer Social Insurance Tax"],
        )
        """

        # Individuals set reservation wages
        self.individuals.ts.reservation_wages.append(
            self.individuals.compute_reservation_wages(
                unemployment_benefits_by_individual=self.central_government.ts.current(
                    "unemployment_benefits_by_individual"
                )[0],
            )
        )

    def clear_labour_market(self) -> None:
        """Execute labor market clearing.

        Matches job seekers with vacancies and determines employment
        and wages through the labor market mechanism.
        """
        logging.info("Clearing labour market for %s", self.country_name)
        labour_costs = self.labour_market.clear(
            firms=self.firms,
            households=self.households,
            individuals=self.individuals,
        )
        self.firms.ts.labour_costs.append(labour_costs)

    def update_planning_metrics(self) -> None:
        """Update forward-looking economic indicators.

        Computes expected profits, asset values, benefits, and other metrics
        used by agents in their planning decisions.
        """
        # Firms estimate profits
        self.firms.ts.expected_profits.append(
            self.firms.compute_estimated_profits(
                estimated_growth=self.economy.ts.current("estimated_growth")[0],
                estimated_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
            )
        )

        # Banks estimate profits
        self.banks.ts.expected_profits.append(
            self.banks.compute_estimated_profits(
                estimated_growth=self.economy.ts.current("estimated_growth")[0],
                estimated_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
            )
        )

        # Firms compute the expected value of its capital stock
        self.firms.ts.expected_capital_inputs_stock_value.append(
            self.firms.compute_expected_capital_inputs_stock_value(
                current_good_prices=self.economy.ts.current("good_prices"),
                estimated_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
            )
        )

        # The central government updates unemployment benefits paid to individuals and social transfers to households
        self.central_government.update_benefits(
            historic_ppi_inflation=self.economy.ts.historic("ppi_inflation"),
            exogenous_ppi_inflation=self.exogenous.inflation_before["PPI Inflation"].values,
            current_estimated_ppi_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
            current_unemployment_rate=self.economy.ts.current("unemployment_rate")[0],
            current_estimated_growth=self.economy.ts.current("estimated_growth")[0],
        )

        # Individuals update their income from unemployment benefits
        self.individuals.ts.income_from_unemployment_benefits.append(
            self.central_government.distribute_unemployment_benefits_to_individuals(
                current_individual_activity_status=self.individuals.states["Activity Status"],
            )
        )

        # Individual labour inputs
        self.individuals.ts.labour_inputs.append(self.individuals.compute_labour_inputs())

        # Central bank policy rate
        self.central_bank.ts.policy_rate.append(
            [
                self.central_bank.compute_rate(
                    inflation=self.economy.ts.current("ppi_inflation")[0],
                    growth=self.economy.ts.current("total_growth")[0],
                )
            ]
        )

        # Number of employees for each firm
        self.firms.ts.number_of_employees.append(
            self.firms.compute_n_employees(
                corresponding_firm=self.individuals.states["Corresponding Firm ID"],
            )
        )
        self.firms.ts.number_of_employees_histogram.append(
            get_histogram(self.firms.ts.current("number_of_employees"), None)
        )
        self.firms.ts.output_by_employee_histogram.append(
            get_histogram(
                self.firms.ts.current("production") / self.firms.ts.current("number_of_employees"),
                None,
            )
        )

        # Firm labour inputs
        labour_inputs_from_employees = self.firms.compute_labour_inputs(
            corresponding_firm=self.individuals.states["Corresponding Firm ID"],
            current_labour_inputs=self.individuals.ts.current("labour_inputs"),
        )

        # Firm wages
        self.individuals.ts.employee_income.append(
            self.firms.set_employee_income(
                corresponding_firm=self.individuals.states["Corresponding Firm ID"],
                current_individual_labour_inputs=self.individuals.ts.current("labour_inputs"),
                current_individual_stating_new_job=self.individuals.states["Started New Job"],
                current_employee_income=self.individuals.ts.current("employee_income"),
                current_individual_offered_wage=self.individuals.states["Offered Wage of Accepted Job"],
                labour_inputs_from_employees=labour_inputs_from_employees,
                estimated_ppi_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
                income_taxes=self.central_government.states["Income Tax"],
                employee_social_insurance_tax=self.central_government.states["Employee Social Insurance Tax"],
                employer_social_insurance_tax=self.central_government.states["Employer Social Insurance Tax"],
            )
        )
        self.individuals.ts.employee_income_histogram.append(
            get_histogram(self.individuals.ts.current("employee_income"), self.scale)
        )

        # Update TFP before production (only if TFP growth is configured)
        if not self.assume_zero_growth:
            self.firms.update_tfp()
            # Update technical coefficient multipliers
            self.firms.update_technical_coefficients()

        # Firm production
        if self.assume_zero_growth:
            self.firms.ts.production.append(self.firms.ts.initial("production"))
        else:
            self.firms.ts.production.append(self.firms.compute_production())
        self.firms.ts.production_histogram.append(get_histogram(self.firms.ts.current("production"), None))

        # Firm prices
        if not self.assume_zero_growth:
            self.firms.ts.price.append(
                self.firms.compute_price(
                    current_estimated_ppi_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
                    previous_average_good_prices=self.economy.ts.current("good_prices"),
                    ppi_during=self.exogenous.national_accounts_during["PPI (Value)"].values.flatten(),
                )
            )

        # Firm demand for goods
        self.firms.ts.unconstrained_target_intermediate_inputs.append(
            self.firms.compute_unconstrained_demand_for_intermediate_inputs(
                good_prices=self.economy.ts.current("good_prices")
            )
        )
        self.firms.ts.unconstrained_target_intermediate_inputs_costs.append(
            self.firms.compute_unconstrained_demand_for_intermediate_inputs_value(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )
        self.firms.ts.unconstrained_target_capital_inputs.append(
            self.firms.compute_unconstrained_demand_for_capital_inputs(
                good_prices=self.economy.ts.current("good_prices")
            )
        )
        self.firms.ts.unconstrained_target_capital_inputs_costs.append(
            self.firms.compute_unconstrained_demand_for_capital_inputs_value(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )

        # Individual income
        self.individuals.ts.expected_income.append(
            self.individuals.compute_expected_income(
                expected_firm_profits=self.firms.ts.current("expected_profits"),
                expected_bank_profits=self.banks.ts.current("expected_profits"),
                cpi=self.economy.ts.current("cpi")[0],
                expected_inflation=self.economy.ts.current("estimated_cpi_inflation")[0],
                income_taxes=self.central_government.states["Income Tax"],
                tau_firm=self.central_government.states["Profit Tax"],
            )
        )

        # Household income
        self.households.ts.expected_income_employee.append(
            self.households.compute_employee_income(
                individual_income=self.individuals.ts.current("expected_income"),
                corr_households=self.individuals.states["Corresponding Household ID"],
            )
        )
        self.households.ts.expected_income_social_transfers.append(
            self.households.compute_expected_social_transfer_income(
                total_other_social_transfers=self.central_government.ts.current("total_other_benefits")[0],
                cpi=self.economy.ts.current("cpi")[0],
                expected_inflation=self.economy.ts.current("estimated_cpi_inflation")[0],
            )
        )
        self.households.ts.income_rental.append(
            self.households.compute_rental_income(
                housing_data=self.housing_market.states["properties"],
                income_taxes=self.central_government.states["Income Tax"],
            )
        )
        self.households.ts.total_income_rental.append([self.households.ts.current("income_rental").sum()])
        self.households.ts.expected_income_financial_assets.append(
            self.households.compute_expected_income_from_financial_assets()
        )
        self.households.ts.expected_income.append(self.households.compute_expected_income())

        # Household target consumption
        # Note: For CES substitution, we track additional taxes (like carbon tax) similar to firms
        # Currently no additional taxes are implemented, so both initial and current are zero
        # This will be updated when carbon tax or other additional taxes are added
        current_additional_taxes = np.zeros(len(self.firms.ts.current("price")))
        initial_additional_taxes = np.zeros(len(self.firms.ts.current("price")))

        self.households.ts.target_consumption.append(
            self.households.compute_target_consumption(
                expected_inflation=self.economy.ts.current("estimated_cpi_inflation")[0],
                current_cpi=self.economy.ts.current("cpi")[0],
                initial_cpi=self.economy.ts.initial("cpi")[0],
                exogenous_total_consumption=self.exogenous.national_accounts_during[
                    "Real Household Consumption (Value)"
                ].values.flatten(),
                per_capita_unemployment_benefits=self.central_government.ts.current(
                    "unemployment_benefits_by_individual"
                )[0],
                tau_vat=self.central_government.states["Value-added Tax"],
                assume_zero_growth=self.assume_zero_growth,
                prices=self.firms.ts.current("price"),
                initial_prices=self.firms.ts.initial("price"),
                taxes=current_additional_taxes,
                initial_taxes=initial_additional_taxes,
            )
        )

        # Household target investment
        self.households.ts.target_investment.append(
            self.households.compute_target_investment(
                expected_inflation=self.economy.ts.current("estimated_cpi_inflation")[0],
                current_cpi=self.economy.ts.current("cpi")[0],
                initial_cpi=self.economy.ts.initial("cpi")[0],
                exogenous_total_investment=self.exogenous.national_accounts_during[
                    "Real Household Investment (Value)"
                ].values.flatten(),
                tau_cf=self.central_government.states["Capital Formation Tax"],
                assume_zero_growth=self.assume_zero_growth,
            )
        )

    def prepare_housing_market_clearing(self) -> None:
        """Prepare for housing market transactions.

        Updates property values and household housing decisions before
        market clearing.
        """
        # Update property values
        self.housing_market.update_property_value()

        # Decide on whether to remain, rent or buy
        self.households.prepare_housing_market_clearing(
            housing_data=self.housing_market.states["properties"],
            observed_fraction_value_price=self.housing_market.ts.current("observed_fraction_value_price"),
            observed_fraction_rent_value=self.housing_market.ts.current("observed_fraction_rent_value"),
            expected_hpi_growth=self.economy.ts.current("estimated_hpi_inflation")[0],
            assumed_mortgage_maturity=self.banks.parameters.mortgage_maturity,
            rental_income_taxes=self.central_government.states["Income Tax"],
        )

        # Set rent
        self.households.update_rent(
            housing_data=self.housing_market.states["properties"],
            historic_inflation=self.economy.ts.historic("cpi_inflation"),
            exogenous_inflation_before=self.exogenous.inflation_before["CPI Inflation"].values,
        )

    def clear_housing_market(self) -> None:
        """Execute housing market clearing.

        Matches buyers/renters with sellers/landlords and determines
        property transactions.
        """
        self.housing_market.clear(
            household_main_residence_tenure_status=self.households.states["Tenure Status of the Main Residence"],
            max_price_willing_to_pay=self.households.ts.current("max_price_willing_to_pay"),
            max_rent_willing_to_pay=self.households.ts.current("max_rent_willing_to_pay"),
        )

    def prepare_credit_market_clearing(self) -> None:
        """Prepare for credit market transactions.

        Updates loan demands and interest rates before market clearing.
        """
        self.firms.compute_target_credit(
            estimated_growth=self.economy.ts.current("estimated_growth")[0],
            estimated_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
        )
        self.households.compute_target_credit(
            current_sales=self.housing_market.states["current_sales"].loc[
                self.housing_market.states["current_sales"]["sales_types"] == "Rental"
            ],
        )
        self.banks.set_interest_rates(central_bank_policy_rate=self.central_bank.ts.current("policy_rate")[0])

    def clear_credit_market(self) -> None:
        """Execute credit market clearing.

        Matches borrowers with lenders and determines loan allocations
        and terms.
        """
        self.credit_market.clear(
            banks=self.banks,
            firms=self.firms,
            households=self.households,
            current_npl_firm_loans=self.economy.ts.current("npl_firm_loans")[0],
            current_npl_hh_cons_loans=self.economy.ts.current("npl_hh_cons_loans")[0],
            current_npl_mortgages=self.economy.ts.current("npl_mortgages")[0],
        )

    def process_housing_market_clearing(self) -> None:
        """Process housing market outcomes.

        Updates property ownership, rents, and related financial positions
        after market clearing.
        """
        self.housing_market.ts.observed_fraction_value_price.append(
            self.housing_market.compute_observed_fraction_value_price()
        )
        self.housing_market.ts.observed_fraction_rent_value.append(
            self.housing_market.compute_observed_fraction_rent_value()
        )
        self.housing_market.process_housing_market_clearing(
            household_states=self.households.states,
            household_received_mortgages=self.households.ts.current("received_mortgages"),
            household_financial_wealth=self.households.ts.current("wealth_financial_assets"),
        )
        self.households.process_housing_market_clearing(
            housing_data=self.housing_market.states["properties"],
            social_housing_function=self.central_government.functions["social_housing"],
            current_sales=self.housing_market.states["current_sales"].loc[
                self.housing_market.states["current_sales"]["sales_types"] == "Sell"
            ],
            current_unemployment_benefits_by_individual=self.central_government.ts.current(
                "unemployment_benefits_by_individual"
            )[0],
        )

    def process_credit_market_clearing(self) -> None:
        """Process credit market outcomes.

        Updates loan positions, interest payments, and related financial
        positions after market clearing.
        """
        # Handle debt installments
        self.firms.ts.debt_installments.append(self.credit_market.pay_firm_installments())
        self.firms.ts.total_debt_installments.append([self.firms.ts.current("debt_installments").sum()])
        self.households.ts.debt_installments.append(self.credit_market.pay_household_installments())
        self.households.ts.total_debt_installments.append([self.households.ts.current("debt_installments").sum()])
        self.credit_market.remove_repaid_loans()

        # Compute aggregates
        self.credit_market.compute_aggregates()

        # Calculate firm debt
        self.firms.ts.short_term_loan_debt.append(self.credit_market.compute_outstanding_short_term_loans_by_firm())
        self.firms.ts.long_term_loan_debt.append(self.credit_market.compute_outstanding_long_term_loans_by_firm())
        self.firms.ts.debt.append(self.firms.compute_debt())

        # Calculate the interest on loans paid by firms
        self.firms.ts.interest_paid_on_loans.append(self.credit_market.compute_interest_paid_by_firm())

        # Calculate the interest on deposits received/paid by firms
        self.firms.ts.interest_paid_on_deposits.append(
            self.firms.compute_interest_paid_on_deposits(
                bank_interest_rate_on_firm_deposits=self.banks.ts.current("interest_rate_on_firm_deposits"),
                bank_overdraft_rate_on_firm_deposits=self.banks.ts.current("overdraft_rate_on_firm_deposits"),
            )
        )

        # Calculate paid interest of firms
        self.firms.ts.interest_paid.append(self.firms.compute_interest_paid())

        # Calculate household debt
        self.households.ts.consumption_loan_debt.append(
            self.credit_market.compute_outstanding_consumption_loans_by_household()
        )
        self.households.ts.mortgage_debt.append(self.credit_market.compute_outstanding_mortgages_by_household())
        self.households.ts.debt.append(self.households.compute_debt())
        self.households.ts.debt_histogram.append(get_histogram(self.households.ts.current("debt"), self.scale))

        # Calculate the interest on loans paid by households
        self.households.ts.interest_paid_on_loans.append(self.credit_market.compute_interest_paid_by_household())

        # Calculate the interest on deposits received/paid by households
        self.households.ts.interest_paid_on_deposits.append(
            self.households.compute_interest_paid_on_deposits(
                bank_interest_rate_on_household_deposits=self.banks.ts.current("interest_rate_on_household_deposits"),
                bank_overdraft_rate_on_household_deposits=self.banks.ts.current("overdraft_rate_on_household_deposits"),
            )
        )

        # Calculate paid interest of households
        self.households.ts.interest_paid.append(self.households.compute_interest_paid())

        # Calculate the interest on loans received by banks
        self.banks.ts.interest_received_on_loans.append(self.credit_market.compute_interest_received_by_bank())

    def prepare_goods_market_clearing(self) -> None:
        """Prepare for goods market transactions.

        Updates production plans, consumption demands, and prices before
        market clearing.
        """
        self.firms.prepare_goods_market_clearing(
            exchange_rate_usd_to_lcu=self.exchange_rate_usd_to_lcu,
            previous_good_prices=self.economy.ts.current("good_prices"),
            expected_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
        )
        self.households.prepare_goods_market_clearing(
            exchange_rate_usd_to_lcu=self.exchange_rate_usd_to_lcu,
        )
        self.government_entities.prepare_goods_market_clearing(
            n_industries=self.economy.n_industries,
            exchange_rate_usd_to_lcu=self.exchange_rate_usd_to_lcu,
            exogenous_gov_consumption_before=(
                None
                if "Real Government Consumption (Value)" not in self.exogenous.national_accounts_before.columns
                else self.exogenous.national_accounts_before["Real Government Consumption (Value)"].values.flatten()
            ),
            exogenous_gov_consumption_during=(
                None
                if "Real Government Consumption (Value)" not in self.exogenous.national_accounts_during.columns
                else self.exogenous.national_accounts_during["Real Government Consumption (Value)"].values.flatten()
            ),
            initial_good_prices=self.economy.ts.initial("good_prices"),
            current_good_prices=self.economy.ts.current("good_prices"),
            historic_ppi=np.array(self.economy.ts.historic("ppi")).flatten(),
            expected_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
            expected_growth=self.economy.ts.current("estimated_growth")[0],
            forecasting_window=self.forecasting_window,
            assume_zero_growth=self.assume_zero_growth,
            assume_zero_noise=self.assume_zero_noise,
        )

    def update_realised_metrics(self) -> None:
        """Update realized economic outcomes after market clearing.

        This method coordinates the comprehensive updating of all economic metrics after markets
        have cleared. It processes the results of market interactions and computes derived
        economic indicators across eight major categories:

        1. Economic Indicators:
           - Price indices (PPI, CPI)
           - Economic growth rates
           - House price indices
           - Labor market metrics (employment, unemployment)
           - Rental market aggregates

        2. Global Trade and Capital Formation:
           - International trade flows
           - Import/export tracking
           - Gross fixed capital formation
           - Input costs (intermediate and capital)

        3. Firm Production and Costs:
           - Production volumes and nominal values
           - Wages and labor costs
           - Inventory management
           - Input utilization
           - Emissions (if enabled)

        4. Firm Financial Metrics:
           - Sales and revenue
           - Profits and taxes
           - Corporate financial positions
           - Insolvency metrics
           - Debt aggregates

        5. Individual and Household Income:
           - Employment income
           - Social transfers
           - Financial asset income
           - Rental income
           - Total household income

        6. Household Wealth and Consumption:
           - Consumption patterns
           - Investment decisions
           - Wealth positions
           - Debt management
           - Insolvency handling

        7. Government and Banking:
           - Tax collection
           - Government revenue and deficit
           - Bank profits and equity
           - Interest payments
           - Market share metrics

        8. GDP and National Accounts:
           - GDP components
           - Value added
           - National income
           - Sectoral balances

        The method ensures consistency between micro-level agent behaviors and macro-level
        economic outcomes, maintaining stock-flow consistency throughout the economy.

        Note:
            This method should be called after all markets (goods, labor, credit, housing)
            have cleared and agents have executed their planned transactions.

        Side Effects:
            Updates numerous time series variables across all economic agents and markets,
            reflecting the realized outcomes of the current period's economic activity.
        """
        # Firms distribute bought goods
        self.firms.distribute_bought_goods()

        # A1. ECONOMIC INDICATORS
        # Update core economic indicators and market metrics
        self.economy.compute_price_indicators(
            firm_real_amount_bought=self.firms.ts.current("real_amount_bought"),
            firm_nominal_amount_spent=self.firms.ts.current("nominal_amount_spent_in_lcu"),
            household_real_amount_bought=self.households.ts.current("real_amount_bought"),
            household_nominal_amount_spent=self.households.ts.current("nominal_amount_spent_in_lcu"),
            government_real_amount_bought=self.government_entities.ts.current("real_amount_bought"),
            government_nominal_amount_spent=self.government_entities.ts.current("nominal_amount_spent_in_lcu"),
            firms_real_amount_bought_as_capital_goods=self.firms.ts.current("real_amount_bought_as_capital_goods"),
        )
        self.economy.compute_inflation()
        self.economy.compute_growth(
            current_production=self.firms.ts.current("production"),
            prev_production=self.firms.ts.prev("production"),
            industries=self.firms.states["Industry"],
        )
        self.economy.compute_house_price_index(
            current_property_values=self.housing_market.ts.current("property_values"),
            previous_property_values=self.housing_market.ts.prev("property_values"),
        )
        self.economy.compute_labour_market_aggregates(
            current_individual_activity_status=self.individuals.states["Activity Status"],
            current_firm_labour_inputs=self.firms.ts.current("labour_inputs"),
            current_desired_firm_labour_inputs=self.firms.ts.current("desired_labour_inputs"),
            num_ind_employed_before_cleaning=self.labour_market.ts.current("num_employed_individuals_before_clearing")[
                0
            ],
            num_ind_newly_joining=self.labour_market.ts.current("num_individuals_newly_joining")[0],
            num_ind_newly_leaving=self.labour_market.ts.current("num_individuals_newly_leaving")[0],
        )
        self.economy.compute_rental_market_aggregates(
            real_rent_paid=self.households.ts.current("rent"),
            imp_rent_paid=self.households.ts.current("rent_imputed"),
            rental_income=self.households.ts.current("income_rental"),
        )

        # B1. GLOBAL TRADE AND CAPITAL FORMATION
        # Record international trade flows and capital formation
        self.economy.record_global_trade(
            firms=self.firms,
            households=self.households,
            government_entities=self.government_entities,
            tau_export=self.central_government.states["Export Tax"],
        )

        # Update gross fixed capital formation
        self.firms.ts.gross_fixed_capital_formation.append(
            self.firms.compute_gross_fixed_capital_formation(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )

        # C1. FIRM PRODUCTION AND COSTS
        # Update input costs and production metrics
        self.firms.update_total_newly_bought_costs(
            current_good_prices=self.economy.ts.current("good_prices"),
        )

        # Execute and record productivity investment after capital purchases are known
        self.firms.execute_productivity_investment()

        self.firms.ts.demand.append(self.firms.compute_demand())

        self.firms.ts.production_nominal.append(
            self.firms.compute_nominal_production(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )

        # C2. WAGES AND LABOR
        self.firms.update_total_wages_paid(
            corresponding_firm=self.individuals.states["Corresponding Firm ID"],
            individual_wages=self.individuals.ts.current("employee_income"),
            income_taxes=self.central_government.states["Income Tax"],
            employee_social_insurance_tax=self.central_government.states["Employee Social Insurance Tax"],
            employer_social_insurance_tax=self.central_government.states["Employer Social Insurance Tax"],
            cpi=self.economy.ts.current("cpi")[0],
        )

        # C3. EMISSIONS AND INVENTORY
        if self.add_emissions:
            readjusted_factors = (
                self.emission_factors_lcu / self.economy.ts.current("good_prices")[self.emitting_indices]
            )
            self.firms.update_emissions(readjusted_factors=readjusted_factors, emitting_indices=self.emitting_indices)

        self.firms.ts.used_intermediate_inputs.append(self.firms.compute_used_intermediate_inputs())
        self.firms.ts.used_intermediate_inputs_costs.append(
            self.firms.compute_used_intermediate_inputs_costs(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )
        self.firms.ts.used_capital_inputs.append(self.firms.compute_used_capital_inputs())
        self.firms.ts.used_capital_inputs_costs.append(
            self.firms.compute_used_capital_inputs_costs(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )
        self.firms.ts.inventory.append(self.firms.compute_inventory())
        self.firms.ts.inventory_nominal.append(
            self.firms.compute_nominal_inventory(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )
        self.firms.ts.intermediate_inputs_stock.append(self.firms.compute_intermediate_inputs_stock())
        self.firms.ts.intermediate_inputs_stock_value.append(
            self.firms.compute_intermediate_inputs_stock_value(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )
        self.firms.ts.intermediate_inputs_stock_industry.append(
            self.firms.ts.current("intermediate_inputs_stock").sum(axis=0)
        )
        self.firms.ts.capital_inputs_stock.append(self.firms.compute_capital_inputs_stock())
        self.firms.ts.capital_inputs_stock_value.append(
            self.firms.compute_intermediate_inputs_stock_value(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )
        self.firms.ts.capital_inputs_stock_industry.append(self.firms.ts.current("capital_inputs_stock").sum(axis=0))

        # D1. FIRM FINANCIAL METRICS
        # Update firm financial positions
        self.firms.ts.total_inventory_change.append(self.firms.compute_total_inventory_change())
        self.firms.ts.total_sales.append(self.firms.compute_total_sales())
        self.firms.ts.taxes_paid_on_production.append(
            self.firms.compute_taxes_paid_on_production(
                taxes_less_subsidies_rates=self.central_government.states["Taxes Less Subsidies Rates"],
            )
        )
        self.firms.ts.profits.append(self.firms.compute_profits())
        self.firms.ts.unit_costs.append(self.firms.compute_unit_costs())
        self.firms.ts.corporate_taxes_paid.append(
            self.firms.compute_corporate_taxes_paid(
                tau_firm=self.central_government.states["Profit Tax"],
            )
        )

        # D2. FIRM BALANCE SHEETS
        self.firms.ts.gross_operating_surplus_mixed_income.append(
            self.firms.compute_gross_operating_surplus_mixed_income()
        )
        self.firms.ts.deposits.append(self.firms.compute_deposits())
        self.firms.ts.equity.append(
            self.firms.compute_equity(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )

        # D3. FIRM INSOLVENCY
        npl_firm_loans = self.firms.handle_insolvency(credit_market=self.credit_market)
        self.economy.ts.npl_firm_loans.append([npl_firm_loans])

        firm_insolvency_rate, num_insolvent_firms_by_sector = self.firms.compute_insolvency_rate()
        self.economy.ts.firm_insolvency_rate.append([firm_insolvency_rate])
        self.economy.ts.num_insolvent_firms_by_sector.append(num_insolvent_firms_by_sector)

        self.firms.ts.total_debt.append([self.firms.compute_total_debt()])
        self.firms.ts.total_deposits.append([self.firms.compute_total_deposits()])

        # E1. INDIVIDUAL AND HOUSEHOLD INCOME
        # Update individual income components
        self.individuals.ts.income.append(
            self.individuals.compute_income(
                firm_profits=self.firms.ts.current("profits"),
                bank_profits=self.banks.ts.current("profits"),
                cpi=self.economy.ts.current("cpi")[0],
                income_taxes=self.central_government.states["Income Tax"],
                tau_firm=self.central_government.states["Profit Tax"],
            )
        )
        self.individuals.ts.income_histogram.append(get_histogram(self.individuals.ts.current("income"), self.scale))

        # E2. HOUSEHOLD INCOME COMPONENTS
        # Recalculate rental income with final housing market data and overwrite the planned value
        final_income_rental = self.households.compute_rental_income(
            housing_data=self.housing_market.states["properties"],
            income_taxes=self.central_government.states["Income Tax"],
        )
        self.households.ts.dicts["income_rental"][-1] = final_income_rental
        self.households.ts.dicts["total_income_rental"][-1] = [final_income_rental.sum()]

        self.households.ts.income_employee.append(
            self.households.compute_employee_income(
                individual_income=self.individuals.ts.current("employee_income"),
                corr_households=self.individuals.states["Corresponding Household ID"],
            )
        )
        self.households.ts.total_income_employee.append([self.households.ts.current("income_employee").sum()])
        self.households.ts.income_social_transfers.append(
            self.households.compute_social_transfer_income(
                total_other_social_transfers=self.central_government.ts.current("total_other_benefits")[0],
                cpi=self.economy.ts.current("cpi")[0],
            )
        )
        self.households.ts.total_income_social_transfers.append(
            [self.households.ts.current("income_social_transfers").sum()]
        )
        self.households.ts.income_financial_assets.append(self.households.compute_income_from_financial_assets())
        self.households.ts.total_income_financial_assets.append(
            [self.households.ts.current("income_financial_assets").sum()]
        )
        self.households.ts.income.append(self.households.compute_income())
        self.households.ts.income_histogram.append(get_histogram(self.households.ts.current("income"), self.scale))

        # E3. HOUSEHOLD METRICS
        rent_div_income = np.divide(
            self.households.ts.current("rent"),
            self.households.ts.current("income"),
            out=np.zeros(self.households.ts.current("rent").shape),
            where=self.households.ts.current("income") != 0.0,
        )
        self.households.ts.rent_div_income_histogram.append(get_histogram(rent_div_income, None))

        # F1. HOUSEHOLD WEALTH AND CONSUMPTION
        # Update household financial positions
        if self.add_emissions:
            readjusted_factors = (
                self.emission_factors_lcu / self.economy.ts.current("good_prices")[self.emitting_indices]
            )
        else:
            readjusted_factors = None

        self.households.update_consumption_and_investment(
            tau_vat=self.central_government.states["Value-added Tax"],
            tau_cf=self.central_government.states["Capital Formation Tax"],
            readjusted_factors=readjusted_factors,
            emitting_indices=self.emitting_indices,
            add_emissions=self.add_emissions,
        )
        self.households.update_wealth(
            housing_data=self.housing_market.states["properties"],
            tau_cf=self.central_government.states["Capital Formation Tax"],
        )
        self.households.ts.wealth_histogram.append(get_histogram(self.households.ts.current("wealth"), self.scale))
        self.households.ts.net_wealth.append(self.households.compute_net_wealth())

        # F2. HOUSEHOLD INSOLVENCY
        household_insolvency_rate, npl_hh_cons_loans, npl_mortgages = self.households.handle_insolvency(
            banks=self.banks,
            credit_market=self.credit_market,
        )
        self.economy.ts.household_insolvency_rate.append([household_insolvency_rate])
        self.economy.ts.npl_hh_cons_loans.append([npl_hh_cons_loans])
        self.economy.ts.npl_mortgages.append([npl_mortgages])

        # G1. GOVERNMENT AND BANKING
        # Update government consumption
        self.government_entities.record_consumption(
            add_emissions=self.add_emissions,
            readjusted_factors=readjusted_factors,
            emitting_indices=self.emitting_indices,
        )

        # G2. BANKING METRICS
        self.banks.ts.interest_received_on_deposits.append(
            self.banks.compute_interest_received_on_deposits(
                central_bank_policy_rate=self.central_bank.ts.current("policy_rate"),
            )
        )
        self.banks.ts.profits.append(self.banks.compute_profits())
        self.banks.ts.profits_histogram.append(get_histogram(self.banks.ts.current("profits"), self.scale))

        # G3. BANK BALANCE SHEETS
        self.banks.update_deposits(
            current_firm_deposits=self.firms.ts.current("deposits"),
            current_household_deposits=self.households.ts.current("wealth_deposits"),
            firm_corresponding_bank=self.firms.states["Corresponding Bank ID"],
            households_corresponding_bank=self.households.states["Corresponding Bank ID"],
        )
        self.banks.update_loans(credit_market=self.credit_market)

        self.banks.ts.market_share.append(self.banks.compute_market_share())
        self.banks.ts.market_share_histogram.append(get_histogram(self.banks.ts.current("market_share"), None))

        self.banks.ts.equity.append(
            self.banks.compute_equity(
                profit_taxes=self.central_government.states["Profit Tax"],
            )
        )
        self.banks.ts.equity_histogram.append(get_histogram(self.banks.ts.current("equity"), self.scale))

        self.banks.ts.liability.append(self.banks.compute_liability())
        self.banks.ts.liability_histogram.append(get_histogram(self.banks.ts.current("liability"), self.scale))

        self.banks.ts.deposits.append(self.banks.compute_deposits())
        self.banks.ts.deposits_histogram.append(get_histogram(self.banks.ts.current("deposits"), self.scale))

        # G4. BANK INSOLVENCY
        self.central_government.ts.bank_equity_injection.append(
            [self.banks.handle_insolvency(credit_market=self.credit_market)]
        )
        self.economy.ts.bank_insolvency_rate.append([self.banks.compute_insolvency_rate()])

        # G5. GOVERNMENT REVENUE
        # General government fields
        self.central_government.compute_taxes(
            current_ind_employee_income=self.individuals.ts.current("employee_income"),
            current_total_rent_paid=self.households.ts.current("rent")[
                self.households.states["Tenure Status of the Main Residence"] == 0
            ].sum(),
            current_income_financial_assets=self.households.ts.current("income_financial_assets"),
            current_ind_activity=self.individuals.states["Activity Status"],
            current_ind_realised_cons=self.households.ts.current("consumption"),
            current_bank_profits=self.banks.ts.current("profits"),
            current_firm_production=self.firms.ts.current("production"),
            current_firm_price=self.firms.ts.current("price"),
            current_firm_profits=self.firms.ts.current("profits"),
            current_firm_industries=self.firms.states["Industry"],
            taxes_less_subsidies_rates=self.central_government.states["Taxes Less Subsidies Rates"],
            current_household_new_real_wealth=self.households.ts.current("investment"),
            current_total_exports=self.economy.ts.current("exports_before_taxes").sum(),
        )

        # General government fields
        self.central_government.ts.taxes_on_products.append([self.central_government.compute_taxes_on_products()])
        self.central_government.ts.revenue.append(
            [
                self.central_government.compute_revenue(
                    household_rent_paid_to_government=self.households.states["Rent paid to Government"]
                )
            ]
        )
        self.central_government.ts.deficit.append(
            self.central_government.compute_deficit(
                current_ind_activity=self.individuals.states["Activity Status"],
                current_household_social_transfers=self.households.ts.current("income_social_transfers"),
                current_government_nominal_amount_spent=self.government_entities.ts.current(
                    "nominal_amount_spent_in_lcu"
                ),
                government_interest_rates=self.central_bank.ts.current("policy_rate")[0],
            )
        )
        self.central_government.ts.debt.append(self.central_government.compute_debt())

        # Compute GDP
        self.economy.compute_gdp(
            total_output=(self.firms.ts.current("price") * self.firms.ts.current("production")).sum(),
            sectoral_sales=np.bincount(
                self.firms.states["Industry"],
                weights=self.firms.ts.current("total_sales"),
                minlength=self.economy.n_industries,
            ),
            sectoral_intermediate_consumption=np.bincount(
                self.firms.states["Industry"],
                weights=self.firms.ts.current("used_intermediate_inputs_costs"),
                minlength=self.economy.n_industries,
            ),
            taxes_on_products=self.central_government.ts.current("taxes_on_products")[0],
            taxes_on_production=self.central_government.ts.current("taxes_production")[0],
            rent_paid=self.economy.ts.current("total_real_rent_paid")[0],
            rent_imputed=self.economy.ts.current("total_imp_rent_paid")[0],
            hh_consumption=self.households.ts.current("total_consumption")[0],
            gov_consumption=self.government_entities.ts.current("total_consumption")[0],
            change_in_inventories=self.firms.ts.current("total_inventory_change").sum()
            + self.firms.ts.current("total_intermediate_inputs_bought_costs").sum()
            - self.firms.ts.current("used_intermediate_inputs_costs").sum(),
            gross_fixed_capital_formation=self.firms.ts.current("total_capital_inputs_bought_costs").sum()
            + (1 + self.central_government.states["Capital Formation Tax"])
            * self.households.ts.current("investment").sum(),
            exports=self.economy.ts.current("exports").sum(),
            imports=self.economy.ts.current("imports").sum(),
            operating_surplus=self.firms.ts.current("gross_operating_surplus_mixed_income").sum(),
            wages=self.firms.ts.current("total_wage").sum(),
            rent_received=self.economy.ts.current("total_real_rent_rec")[0]
            + self.central_government.ts.current("total_rent_received")[0]
            + self.central_government.ts.current("taxes_rental_income")[0],
            running_multiple_countries=self.running_multiple_countries,
        )

    def update_population_structure(self) -> None:
        """Update demographic composition.

        Applies demographic changes (aging, retirement, etc.) to the
        population of individual agents.
        """
        self.individuals.update_demography()

    def get_goods_market_participants(self) -> list[Agent | RestOfTheWorld]:
        """Get all participants in the goods market.

        Returns:
            list[Agent | RestOfTheWorld]: List of agents participating in
                goods market transactions
        """
        return [self.firms, self.households, self.government_entities]

    def save_to_h5(self, h5_file: h5py.File) -> None:
        """Save complete country state to HDF5.

        Saves the full state of all agents, markets, and economic variables
        to an HDF5 file for later analysis or continuation.

        Args:
            h5_file (h5py.File): Open HDF5 file to save to
        """
        group = h5_file.create_group(self.country_name)
        self.firms.save_to_h5(group)
        # self.firms.save_industry_firms_df(group)

        self.individuals.save_to_h5(group)
        self.households.save_to_h5(group)
        self.households.save_consumption_weights(group)
        self.government_entities.save_to_h5(group)
        self.central_government.save_to_h5(group)
        self.banks.save_to_h5(group)
        self.central_bank.save_to_h5(group)
        self.economy.save_to_h5(group)

        self.labour_market.save_to_h5(group)
        self.credit_market.save_to_h5(group)
        self.housing_market.save_to_h5(group)

        self.exogenous.save_to_h5(group)

    def shallow_output(self) -> pd.DataFrame:
        """Create summary DataFrame of key economic indicators.

        Returns:
            pd.DataFrame: DataFrame containing main economic metrics
                (GDP, inflation, unemployment, etc.)
        """
        data_dict = {
            "Sales": self.firms.total_sales(),
            "Production": self.firms.total_production(),
            "Taxes Paid on Production": self.firms.total_taxes_paid_on_production(),
            "Used Input Costs": self.firms.total_used_input_costs(),
            "Bought Input Costs": self.firms.total_bought_input_costs(),
            "Operating Surplus": self.firms.total_operating_surplus(),
            "Wages": self.firms.total_wages(),
            "Inventory Changes": self.firms.total_inventory_change(),
            "Profits": self.firms.total_profits(),
            "Capital Bought": self.firms.total_capital_bought(),
            "Household Consumption": self.households.total_consumption(),
            "Government Consumption": self.government_entities.total_consumption(),
            "Taxes on Products": self.central_government.total_taxes(),
            "Imports": self.economy.total_imports(),
            "Exports": self.economy.total_exports(),
            "PPI": self.economy.total_ppi_inflation(),
            "CPI": self.economy.total_cpi_inflation(),
            "CFPI": self.economy.total_cfpi_inflation(),
            "Gross Output": self.firms.total_sales() + self.firms.total_taxes_paid_on_production(),
            "Unemployment Rate": self.economy.unemployment_rate(),
            "Consumption Expansion Loan Debt": self.households.consumption_loan_debt(),
            "Mortgage Debt": self.households.mortgage_debt(),
            "Central Bank Policy Rate": self.central_bank.ts.get_aggregate("policy_rate"),
        }

        if self.add_emissions:
            data_dict["Firm Input Emissions"] = self.firms.get_total_inputs_emissions()
            data_dict["Firm Capital Emissions"] = self.firms.get_total_capital_emissions()
            data_dict["Household Consumption Emissions"] = self.households.consumption_emissions()
            data_dict["Household Investment Emissions"] = self.households.investment_emissions()
            data_dict["Government Emissions"] = self.government_entities.emissions()
            for input_name in ["Coal", "Gas", "Oil", "Refined Products"]:
                input_rename = input_name.replace(" ", "_").lower()
                data_dict[f"Firm Input Emissions {input_name}"] = self.firms.get_disaggregated_input_emissions(
                    input_rename
                )
                data_dict[f"Firm Capital Emissions {input_name}"] = self.firms.get_disaggregated_capital_emissions(
                    input_rename
                )
                data_dict[f"Household Consumption Emissions {input_name}"] = (
                    self.households.disaggregated_consumption_emissions(input_rename)
                )
                data_dict[f"Household Investment Emissions {input_name}"] = (
                    self.households.disaggregated_investment_emissions(input_rename)
                )
                data_dict[f"Government Emissions {input_name}"] = self.government_entities.disaggregated_emissions(
                    input_rename
                )

        return pd.DataFrame(data_dict)

    @property
    def n_individuals(self) -> int:
        """int: Total number of individual agents in the economy."""
        return self.individuals.n_individuals
