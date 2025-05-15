"""Household economic agent implementation.

This module implements household economic behavior through:
- Consumption and investment decisions
- Income generation and management
- Wealth accumulation and allocation
- Housing market participation
- Credit market interactions

The implementation handles:
- Household demographics
- Income sources (employment, transfers, rental, financial)
- Consumption patterns
- Investment decisions
- Property ownership
- Financial assets/liabilities
- Debt management
"""

import warnings
from typing import Any, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

from macro_data import SyntheticCountry, SyntheticPopulation
from macromodel.agents.agent import Agent
from macromodel.agents.banks.banks import Banks
from macromodel.agents.households.household_properties import HouseholdType
from macromodel.agents.households.households_ts import create_households_timeseries
from macromodel.configurations import HouseholdsConfiguration
from macromodel.markets.credit_market.credit_market import CreditMarket
from macromodel.markets.goods_market.value_type import ValueType
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import functions_from_model, update_functions
from macromodel.util.get_histogram import get_histogram
from macromodel.util.property_mapping import map_to_enum


class Households(Agent):
    """Economic agent representing household sector behavior.

    This class implements household economic decisions through:
    - Income generation (employment, transfers, rental, financial)
    - Consumption allocation across industries
    - Investment in real and financial assets
    - Housing market participation (buying, renting)
    - Credit market interactions (mortgages, loans)
    - Wealth management and allocation

    The implementation considers:
    - Household demographics and composition
    - Income sources and distribution
    - Consumption patterns and preferences
    - Investment strategies
    - Property ownership and rental decisions
    - Financial asset holdings
    - Debt levels and servicing

    Attributes:
        functions (dict): Mapping of function names to implementations
        independents (list): Independent variables for calculations
        consumption_weights (np.ndarray): Industry-specific consumption shares
        consumption_weights_by_income (np.ndarray): Income-based consumption patterns
        investment_weights (np.ndarray): Industry-specific investment shares
        use_consumption_weights_by_income (bool): Whether to use income-based weights
    """

    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        functions: dict[str, Any],
        ts: TimeSeries,
        states: dict[str, float | np.ndarray | list[np.ndarray]],
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        investment_weights: np.ndarray,
        use_consumption_weights_by_income: bool,
        independents: list[str],
    ):
        """Initialize household economic agent.

        Args:
            country_name (str): Name of the country
            all_country_names (list[str]): List of all country names
            n_industries (int): Number of industries in the economy
            functions (dict[str, Any]): Function implementations for household behavior
            ts (TimeSeries): Time series data container
            states (dict): State variables and parameters
            consumption_weights (np.ndarray): Industry-specific consumption shares
            consumption_weights_by_income (np.ndarray): Income-based consumption patterns
            investment_weights (np.ndarray): Industry-specific investment shares
            use_consumption_weights_by_income (bool): Whether to use income-based weights
            independents (list[str]): Independent variables for calculations
        """
        n_entities = ts.current("n_households")
        super().__init__(
            country_name,
            all_country_names,
            n_industries,
            n_entities,
            n_entities,
            ts,
            states,
            transactor_settings={
                "Buyer Value Type": ValueType.NOMINAL,
                "Seller Value Type": ValueType.NONE,
                "Buyer Priority": 0,
                "Seller Priority": 0,
            },
        )

        self.functions = functions

        self.independents = independents

        # Set initial values
        self.ts["saving_rates_histogram"] = get_histogram(self.get_saving_rates_by_household(), None)

        self.consumption_weights = consumption_weights
        self.consumption_weights_by_income = consumption_weights_by_income.astype(float)

        self.investment_weights = investment_weights

        self.use_consumption_weights_by_income = use_consumption_weights_by_income

    @classmethod
    def from_pickled_agent(
        cls,
        synthetic_population: SyntheticPopulation,
        synthetic_country: SyntheticCountry,
        configuration: HouseholdsConfiguration,
        country_name: str,
        all_country_names: list[str],
        industries: list[str],
        initial_consumption_by_industry: np.ndarray,
        value_added_tax: float,
        scale: int,
        add_emissions: bool = False,
    ) -> "Households":
        """Create household agent from synthetic data.

        Initializes household agent using:
        - Synthetic population demographics
        - Country-specific parameters
        - Initial consumption/investment patterns
        - Tax rates and scaling factors

        Args:
            synthetic_population (SyntheticPopulation): Synthetic household data
            synthetic_country (SyntheticCountry): Country-specific parameters
            configuration (HouseholdsConfiguration): Household behavior config
            country_name (str): Name of the country
            all_country_names (list[str]): List of all country names
            industries (list[str]): List of industry names
            initial_consumption_by_industry (np.ndarray): Initial consumption
            value_added_tax (float): VAT rate
            scale (int): Scaling factor for histograms
            add_emissions (bool): Whether to track emissions

        Returns:
            Households: Initialized household agent
        """
        individual_ages = synthetic_population.individual_data["Age"].values

        corr_individuals = synthetic_population.household_data["Corresponding Individuals ID"]
        corr_individuals = corr_individuals.rename_axis("Household ID")

        corr_renters = synthetic_population.household_data["Corresponding Renters"]
        corr_renters = corr_renters.rename_axis("Household ID")

        corr_owned_houses = synthetic_population.household_data["Corresponding Additionally Owned Houses ID"]

        functions = functions_from_model(model=configuration.functions, loc="macromodel.agents.households")

        hh_data = (
            synthetic_population.household_data.drop(
                columns=[
                    "Corresponding Individuals ID",
                    "Corresponding Renters",
                    "Corresponding Additionally Owned Houses ID",
                ]
            )
            .astype(float)
            .rename_axis("Household ID")
        )

        consumption_weights = synthetic_population.consumption_weights

        consumption_weights_by_income = synthetic_population.consumption_weights_by_income.T

        investment_weights = synthetic_population.investment_weights

        # Additional states
        states: dict[str, float | np.ndarray | list[np.ndarray] | Any] = {
            "saving_rates_model": synthetic_population.saving_rates_model,
            "social_transfers_model": synthetic_population.social_transfers_model,
            "wealth_distribution_model": synthetic_population.wealth_distribution_model,
            "average_saving_rate": synthetic_population.household_data["Saving Rate"].mean(),
            "coefficient_fa_income": synthetic_population.coefficient_fa_income,
            "investment_rate": synthetic_population.household_data["Investment Rate"].values,
        }

        # Additional states
        for state_name in [
            "Type",
            "Corresponding Bank ID",
            "Corresponding Inhabited House ID",
            "Corresponding Property Owner",
            "Tenure Status of the Main Residence",
        ]:
            if state_name not in hh_data.columns:
                raise ValueError(f"Missing {state_name} from the data for initialising households.")
            if state_name == "Type":
                states[state_name] = hh_data[state_name].values.flatten()
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter(action="ignore", category=RuntimeWarning)
                    states[state_name] = hh_data[state_name].fillna(-1).values.astype(int).flatten()
                    states[state_name][states[state_name] < 0] = -1

        # TODO: this is set to 0.2 in Sam's code, and transformed somehow into 0.0945. by the time the data is exported
        #  We need to 1. make this a parameters, 2. move this to the macro-data package.
        #  In general, we should think of where to put the piece of code below.

        investment_rate = synthetic_population.household_data["Investment Rate"].values
        # investment_weights = synthetic_country.industry_data["industry_vectors"]["Household Capital Inputs in LCU"]
        # investment_weights = investment_weights.values / investment_weights.values.sum()
        tau_cf = synthetic_country.tax_data.capital_formation_tax
        income = synthetic_population.household_data["Income"].values  # Income is different from Sam's

        initial_investment = pd.DataFrame(
            data=(1.0 / (1 + tau_cf) * np.outer(investment_weights, investment_rate * income).T),
            index=pd.Index(range(len(synthetic_population.household_data))),
            columns=pd.Index(synthetic_country.industries, name="Industry"),
        )

        tau_vat = synthetic_country.tax_data.value_added_tax

        consumption_by_industry_hh = 1 / (1 + tau_vat) * synthetic_population.industry_consumption_before_vat

        if add_emissions:
            consumption_emissions = synthetic_population.household_data["Consumption Emissions"].values
            investment_emissions = synthetic_population.household_data["Investment Emissions"].values
            coal_consumption_emissions = synthetic_population.household_data["Coal Consumption Emissions"].values
            gas_consumption_emissions = synthetic_population.household_data["Gas Consumption Emissions"].values
            oil_consumption_emissions = synthetic_population.household_data["Oil Consumption Emissions"].values
            refined_products_consumption_emissions = synthetic_population.household_data[
                "Refined Products Consumption Emissions"
            ].values
            coal_investment_emissions = synthetic_population.household_data["Coal Investment Emissions"].values
            gas_investment_emissions = synthetic_population.household_data["Gas Investment Emissions"].values
            oil_investment_emissions = synthetic_population.household_data["Oil Investment Emissions"].values
            refined_products_investment_emissions = synthetic_population.household_data[
                "Refined Products Investment Emissions"
            ].values
        else:
            consumption_emissions = None
            investment_emissions = None
            coal_consumption_emissions = None
            gas_consumption_emissions = None
            oil_consumption_emissions = None
            refined_products_consumption_emissions = None
            coal_investment_emissions = None
            gas_investment_emissions = None
            oil_investment_emissions = None
            refined_products_investment_emissions = None

        ts = create_households_timeseries(
            data=hh_data,
            initial_consumption_by_industry=initial_consumption_by_industry,
            initial_hh_investment=initial_investment.values,
            initial_investment_by_industry=synthetic_population.investment,
            initial_hh_consumption=consumption_by_industry_hh,
            scale=scale,
            vat=value_added_tax,
            tau_cf=tau_cf,
            consumption_emissions=consumption_emissions,
            investment_emissions=investment_emissions,
            coal_consumption_emissions=coal_consumption_emissions,
            gas_consumption_emissions=gas_consumption_emissions,
            oil_consumption_emissions=oil_consumption_emissions,
            refined_products_consumption_emissions=refined_products_consumption_emissions,
            coal_investment_emissions=coal_investment_emissions,
            gas_investment_emissions=gas_investment_emissions,
            oil_investment_emissions=oil_investment_emissions,
            refined_products_investment_emissions=refined_products_investment_emissions,
        )

        # Update the household type
        states["Type"] = map_to_enum(states["Type"], HouseholdType)

        # Corresponding individuals
        states["corr_individuals"] = list(corr_individuals.values)

        # Number of adults individuals in the household
        states["Number of Adults"] = np.array(
            [
                np.sum(individual_ages[states["corr_individuals"][hh_id]] >= 18)
                for hh_id in range(ts.current("n_households"))
            ]
        )

        # Corresponding renters
        states["corr_renters"] = [[int(x) for x in sublist if not pd.isna(x)] for sublist in corr_renters]

        use_consumption_weights_by_income = configuration.take_consumption_weights_by_income_quantile

        independents = configuration.functions.saving_rates.parameters["independents"]

        # TODO: corresponding additionally owned houses is not used

        return cls(
            country_name,
            all_country_names,
            len(industries),
            functions,
            ts,
            states,
            consumption_weights,
            consumption_weights_by_income,
            investment_weights,
            use_consumption_weights_by_income,
            independents,
        )

    def reset(self, configuration: HouseholdsConfiguration) -> None:
        """Reset household agent to initial state.

        Updates function implementations based on new configuration.

        Args:
            configuration (HouseholdsConfiguration): New household config
        """
        self.gen_reset()
        update_functions(functions=self.functions, model=configuration.functions, loc="macromodel.agents.households")

    def compute_employee_income(
        self,
        individual_income: np.ndarray,
        corr_households: np.ndarray,
    ) -> np.ndarray:
        """Calculate household income from employment.

        Aggregates individual employment income to household level.

        Args:
            individual_income (np.ndarray): Income by individual
            corr_households (np.ndarray): Individual-household mapping

        Returns:
            np.ndarray: Employment income by household
        """
        return np.bincount(
            corr_households,
            weights=individual_income,
            minlength=self.ts.current("n_households"),
        )

    def compute_expected_social_transfer_income(
        self,
        total_other_social_transfers: float,
        cpi: float,
        expected_inflation: float,
    ) -> np.ndarray:
        """Calculate expected social transfer income.

        Computes expected transfers based on:
        - Total transfer budget
        - Price level changes
        - Inflation expectations

        Args:
            total_other_social_transfers (float): Total transfer budget
            cpi (float): Current price index
            expected_inflation (float): Expected inflation rate

        Returns:
            np.ndarray: Expected transfers by household
        """
        inds = self.independents
        return (
            (1 + expected_inflation)
            * cpi
            * self.functions["social_transfers"].get_social_transfers(
                n_households=self.ts.current("n_households"),
                total_other_social_transfers=total_other_social_transfers,
                current_independents=(
                    np.array([])
                    if len(inds) == 0
                    else np.stack(
                        [self.ts.current(ind.lower()) for ind in inds],
                        axis=1,
                    )
                ),
                initial_independents=(
                    np.array([])
                    if len(inds) == 0
                    else np.stack(
                        [self.ts.initial(ind.lower()) for ind in inds],
                        axis=1,
                    )
                ),
                model=self.states["social_transfers_model"],
            )
        )

    def compute_social_transfer_income(
        self,
        total_other_social_transfers: float,
        cpi: float,
    ) -> np.ndarray:
        """Calculate current social transfer income.

        Computes actual transfers based on:
        - Total transfer budget
        - Current price level

        Args:
            total_other_social_transfers (float): Total transfer budget
            cpi (float): Current price index

        Returns:
            np.ndarray: Current transfers by household
        """
        inds = self.independents
        return cpi * self.functions["social_transfers"].get_social_transfers(
            n_households=self.ts.current("n_households"),
            total_other_social_transfers=total_other_social_transfers,
            current_independents=(
                np.array([])
                if len(inds) == 0
                else np.stack(
                    [self.ts.current(ind.lower()) for ind in inds],
                    axis=1,
                )
            ),
            initial_independents=(
                np.array([])
                if len(inds) == 0
                else np.stack(
                    [self.ts.initial(ind.lower()) for ind in inds],
                    axis=1,
                )
            ),
            model=self.states["social_transfers_model"],
        )

    def compute_rental_income(
        self,
        housing_data: pd.DataFrame,
        income_taxes: float,
    ) -> np.ndarray:
        """Calculate rental income from property ownership.

        Computes after-tax rental income from:
        - Rented properties
        - Current rent levels
        - Tax rates

        Args:
            housing_data (pd.DataFrame): Property market data
            income_taxes (float): Income tax rate

        Returns:
            np.ndarray: Rental income by household
        """
        housing_data_rented_out = housing_data.loc[
            np.logical_and(
                housing_data["Is Owner-Occupied"] == 0,
                housing_data["Corresponding Inhabitant Household ID"] != -1,
            )
        ]
        housing_data_rented_out_grouped = housing_data_rented_out.groupby("Corresponding Owner Household ID")[
            "Rent"
        ].sum()
        rental_income = np.zeros(self.ts.current("n_households"))
        rental_income[housing_data_rented_out_grouped.index.values] = (
            1 - income_taxes
        ) * housing_data_rented_out_grouped.values
        return rental_income

    def compute_expected_income_from_financial_assets(self) -> np.ndarray:
        """Calculate expected income from financial assets.

        Estimates future financial income based on:
        - Asset holdings
        - Return coefficients
        - Historical patterns

        Returns:
            np.ndarray: Expected financial income by household
        """
        return self.functions["financial_assets"].compute_expected_income(
            income_coefficient=self.states["coefficient_fa_income"],
            initial_other_financial_assets=self.ts.initial("wealth_other_financial_assets"),
            current_other_financial_assets=self.ts.current("wealth_other_financial_assets"),
        )

    def compute_income_from_financial_assets(self) -> np.ndarray:
        """Calculate current income from financial assets.

        Computes actual financial income based on:
        - Current asset holdings
        - Realized returns
        - Income coefficients

        Returns:
            np.ndarray: Current financial income by household
        """
        return self.functions["financial_assets"].compute_income(
            income_coefficient=self.states["coefficient_fa_income"],
            initial_other_financial_assets=self.ts.initial("wealth_other_financial_assets"),
            current_other_financial_assets=self.ts.current("wealth_other_financial_assets"),
        )

    def compute_expected_income(self) -> np.ndarray:
        """Calculate total expected income.

        Aggregates expected income from all sources:
        - Employment
        - Social transfers
        - Rental income
        - Financial assets

        Returns:
            np.ndarray: Total expected income by household
        """
        return (
            self.ts.current("expected_income_employee")
            + self.ts.current("expected_income_social_transfers")
            + self.ts.current("income_rental")
            + self.ts.current("expected_income_financial_assets")
        )

    def compute_income(self) -> np.ndarray:
        """Calculate total current income.

        Aggregates current income from all sources:
        - Employment
        - Social transfers
        - Rental income
        - Financial assets

        Returns:
            np.ndarray: Total current income by household
        """
        return (
            self.ts.current("income_employee")
            + self.ts.current("income_social_transfers")
            + self.ts.current("income_rental")
            + self.ts.current("income_financial_assets")
        )

    def get_saving_rates_by_household(self) -> np.ndarray:
        """Calculate household-specific saving rates.

        Determines saving rates based on:
        - Average saving behavior
        - Household characteristics
        - Economic conditions

        Returns:
            np.ndarray: Saving rates by household
        """
        inds = self.independents
        if len(inds) > 0:
            current_independents = np.stack(
                [self.ts.current(ind.lower()) for ind in inds],
                axis=1,
            )
            initial_independents = np.stack(
                [self.ts.initial(ind.lower()) for ind in inds],
                axis=1,
            )
        else:
            current_independents = np.array([])
            initial_independents = np.array([])
        return self.functions["saving_rates"].get_saving_rates(
            n_households=self.ts.current("n_households"),
            average_saving_rate=self.states["average_saving_rate"],
            current_independents=current_independents,
            initial_independents=initial_independents,
            model=self.states["saving_rates_model"],
        )

    def compute_target_consumption(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        exogenous_total_consumption: float,
        per_capita_unemployment_benefits: float,
        tau_vat: float,
        assume_zero_growth: bool,
    ) -> np.ndarray:
        """Calculate target consumption levels.

        Determines desired consumption based on:
        - Income and saving rates
        - Price level changes
        - Benefit levels
        - Growth assumptions
        - Tax rates

        Args:
            expected_inflation (float): Expected inflation rate
            current_cpi (float): Current price index
            initial_cpi (float): Initial price index
            exogenous_total_consumption (float): External consumption target
            per_capita_unemployment_benefits (float): Per person benefits
            tau_vat (float): Value added tax rate
            assume_zero_growth (bool): Whether to assume no growth

        Returns:
            np.ndarray: Target consumption by household
        """
        saving_rates = self.get_saving_rates_by_household()
        self.ts.saving_rates_histogram.append(get_histogram(saving_rates, None))

        # Target consumption
        if assume_zero_growth:
            return np.outer(
                self.ts.initial("consumption"),
                self.states["consumption_weights_data"],
            ).astype(float)
        else:
            return self.functions["consumption"].compute_target_consumption(
                expected_inflation=expected_inflation,
                current_cpi=current_cpi,
                initial_cpi=initial_cpi,
                historic_consumption_sum=np.array(self.ts.historic("consumption")),
                saving_rates=saving_rates,
                income=self.ts.current("expected_income"),
                household_benefits=self.states["Number of Adults"] * per_capita_unemployment_benefits
                + self.ts.current("expected_income_social_transfers"),
                consumption_weights=self.consumption_weights,
                consumption_weights_by_income=self.consumption_weights_by_income,
                exogenous_total_consumption=exogenous_total_consumption,
                current_time=len(self.ts.historic("total_consumption")),
                take_consumption_weights_by_income_quantile=self.use_consumption_weights_by_income,
                tau_vat=tau_vat,
            )

    def compute_target_investment(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        exogenous_total_investment: float,
        tau_cf: float,
        assume_zero_growth: bool,
    ) -> np.ndarray:
        """Calculate target investment levels.

        Determines desired investment based on:
        - Income and investment rates
        - Price level changes
        - External targets
        - Growth assumptions
        - Tax rates

        Args:
            expected_inflation (float): Expected inflation rate
            current_cpi (float): Current price index
            initial_cpi (float): Initial price index
            exogenous_total_investment (float): External investment target
            tau_cf (float): Capital formation tax rate
            assume_zero_growth (bool): Whether to assume no growth

        Returns:
            np.ndarray: Target investment by household
        """
        if assume_zero_growth:
            return self.ts.initial("investment").astype(float)
        else:
            return self.functions["investment"].compute_target_investment(
                expected_inflation=expected_inflation,
                current_cpi=current_cpi,
                initial_cpi=initial_cpi,
                income=self.ts.current("expected_income"),
                exogenous_total_investment=exogenous_total_investment,
                current_time=len(self.ts.historic("total_investment")),
                investment_weights=self.investment_weights,
                investment_rate=self.states["investment_rate"],
                tau_cf=tau_cf,
            )

    def prepare_housing_market_clearing(
        self,
        housing_data: pd.DataFrame,
        observed_fraction_value_price: np.ndarray,
        observed_fraction_rent_value: np.ndarray,
        expected_hpi_growth: float,
        assumed_mortgage_maturity: int,
        rental_income_taxes: float,
    ) -> None:
        """Prepare for housing market clearing.

        Sets up housing market participation through:
        - Property demand decisions
        - Price/rent willingness
        - Sale listings
        - Rental offerings

        Args:
            housing_data (pd.DataFrame): Property market data
            observed_fraction_value_price (np.ndarray): Price/value ratios
            observed_fraction_rent_value (np.ndarray): Rent/value ratios
            expected_hpi_growth (float): Expected house price growth
            assumed_mortgage_maturity (int): Mortgage term length
            rental_income_taxes (float): Tax rate on rental income
        """
        if len(housing_data) == 0:
            return

        # Households make decisions on their demand for properties
        (
            max_price_willing_to_pay,
            max_rent_willing_to_pay,
            households_hoping_to_move,
        ) = self.functions["property"].compute_demand(
            housing_data=housing_data,
            household_residence_tenure_status=self.states["Tenure Status of the Main Residence"],
            household_income=self.ts.current("expected_income"),
            household_financial_wealth=self.ts.current("wealth_financial_assets"),
            observed_fraction_value_price=observed_fraction_value_price,
            observed_fraction_rent_value=observed_fraction_rent_value,
            expected_hpi_growth=expected_hpi_growth,
            assumed_mortgage_maturity=assumed_mortgage_maturity,
            rental_income_taxes=rental_income_taxes,
        )
        self.ts.max_price_willing_to_pay.append(max_price_willing_to_pay)
        self.ts.max_rent_willing_to_pay.append(max_rent_willing_to_pay)

        # Set price of properties of households that are hoping to move
        ind_mhr_temp_sale = housing_data["Corresponding Owner Household ID"].isin(households_hoping_to_move)
        housing_data.loc[np.logical_not(ind_mhr_temp_sale), "Sale Price"] = np.nan
        ind_still_on_sale = housing_data["Temporarily for Sale"].copy()
        housing_data["Temporarily for Sale"] = False
        housing_data.loc[ind_mhr_temp_sale, "Temporarily for Sale"] = True
        housing_data.loc[
            np.logical_and(ind_mhr_temp_sale, np.logical_not(ind_still_on_sale)),
            "Sale Price",
        ] = self.functions["property"].compute_initial_sale_price(
            property_values=housing_data.loc[
                np.logical_and(ind_mhr_temp_sale, np.logical_not(ind_still_on_sale)),
                "Value",
            ],
        )
        housing_data.loc[np.logical_and(ind_mhr_temp_sale, ind_still_on_sale), "Sale Price"] = self.functions[
            "property"
        ].compute_updated_sale_price(
            sale_prices=housing_data.loc[
                np.logical_and(ind_mhr_temp_sale, ind_still_on_sale),
                "Sale Price",
            ],
        )

        # Set what's up for rent
        prev_up_for_rent = housing_data["Up for Rent"].values
        now_up_for_rent = np.where(np.isnan(housing_data["Corresponding Inhabitant Household ID"].values))[0]
        newly_up_for_rent = [ind for ind in now_up_for_rent if ind not in prev_up_for_rent]
        housing_data["Up for Rent"] = False
        housing_data.loc[now_up_for_rent, "Up for Rent"] = True
        housing_data["Newly on the Rental Market"] = False
        housing_data.loc[newly_up_for_rent, "Newly on the Rental Market"] = True
        not_newly_up_for_rent = np.logical_and(
            np.logical_not(housing_data["Newly on the Rental Market"]),
            housing_data["Up for Rent"],
        )

        # Calculate rent
        housing_data.loc[housing_data["Newly on the Rental Market"], "Rent"] = self.functions[
            "property"
        ].compute_offered_rent_for_new_properties(
            property_value=housing_data.loc[housing_data["Newly on the Rental Market"], "Value"].values,
            observed_fraction_rent_value=observed_fraction_rent_value,
        )
        housing_data.loc[not_newly_up_for_rent, "Rent"] = self.functions[
            "property"
        ].compute_offered_rent_for_existing_properties(
            current_offered_rent=housing_data.loc[not_newly_up_for_rent, "Rent"].values,
        )

    def update_rent(
        self,
        housing_data: pd.DataFrame,
        historic_inflation: list[np.ndarray],
        exogenous_inflation_before: np.ndarray,
    ) -> None:
        """Update rental prices.

        Adjusts rents based on:
        - Historical inflation
        - External price changes
        - Market conditions

        Args:
            housing_data (pd.DataFrame): Property market data
            historic_inflation (list[np.ndarray]): Past inflation rates
            exogenous_inflation_before (np.ndarray): External inflation
        """
        housing_data["Rent"] = self.functions["property"].compute_rent(
            current_rent=housing_data["Rent"].values,
            historic_inflation=np.concatenate(
                (
                    exogenous_inflation_before,
                    np.array(historic_inflation).flatten(),
                )
            ),
        )

    def process_housing_market_clearing(
        self,
        housing_data: pd.DataFrame,
        social_housing_function: Any,
        current_sales: pd.DataFrame,
        current_unemployment_benefits_by_individual: float,
    ) -> None:
        """Process housing market clearing results.

        Updates housing market outcomes:
        - Rent payments
        - Property purchases
        - Social housing allocation
        - Market transactions

        Args:
            housing_data (pd.DataFrame): Property market data
            social_housing_function (Any): Social housing allocator
            current_sales (pd.DataFrame): Market transactions
            current_unemployment_benefits_by_individual (float): Benefits
        """
        # Calculate rent
        rent_by_household, imputed_rent_by_household = self.compute_rent(
            housing_data=housing_data,
            social_housing_function=social_housing_function,
            current_unemployment_benefits_by_individual=current_unemployment_benefits_by_individual,
        )
        self.ts.rent.append(rent_by_household)
        self.ts.rent_imputed.append(imputed_rent_by_household)

        # Calculate the price paid for property
        price_paid_for_property = np.zeros(self.ts.current("n_households"))
        if len(current_sales) > 0:
            price_paid_for_property[current_sales["buyer_id"].values] = current_sales["price_or_rent"].values
        self.ts.price_paid_for_property.append(price_paid_for_property)

    def compute_rent(
        self,
        housing_data: pd.DataFrame,
        social_housing_function: Any,
        current_unemployment_benefits_by_individual: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate rent payments and imputed rent.

        Determines:
        - Actual rent for renters
        - Social housing rent
        - Imputed rent for owners

        Args:
            housing_data (pd.DataFrame): Property market data
            social_housing_function (Any): Social housing allocator
            current_unemployment_benefits_by_individual (float): Benefits

        Returns:
            tuple[np.ndarray, np.ndarray]: Rent paid and imputed rent
        """
        rent_by_household = np.zeros(self.ts.current("n_households"))
        imputed_rent_by_household = np.zeros(self.ts.current("n_households"))

        # Households in social housing
        ind_social_housing = np.where(self.states["Corresponding Inhabited House ID"] == -1)[0]
        social_housing_rent = social_housing_function.compute_social_housing_rent(
            current_unemployment_benefits_by_individual=current_unemployment_benefits_by_individual,
            current_household_size=self.states["Number of Adults"][ind_social_housing],
        )
        rent_by_household[ind_social_housing] = social_housing_rent
        self.states["Rent paid to Government"] = social_housing_rent.sum()

        # Households renting
        ind_renting = self.states["Tenure Status of the Main Residence"] == 3
        rent = housing_data.loc[
            self.states["Corresponding Inhabited House ID"][ind_renting],
            "Rent",
        ].values
        rent_by_household[ind_renting] = rent

        # Households owning
        ind_owning = np.isin(self.states["Tenure Status of the Main Residence"], [1, 2, 4])
        rent = housing_data.loc[
            self.states["Corresponding Inhabited House ID"][ind_owning],
            "Rent",
        ].values
        imputed_rent_by_household[ind_owning] = rent

        return rent_by_household, imputed_rent_by_household

    def compute_target_credit(self, current_sales: pd.DataFrame) -> None:
        """Calculate target credit demand.

        Determines credit needs for:
        - Consumption financing
        - Property purchases
        - Debt management

        Args:
            current_sales (pd.DataFrame): Property transactions
        """
        # Target consumption loans to cover immediate financing gaps
        self.ts.target_consumption_loans.append(
            self.functions["target_credit"].compute_target_consumption_loans(
                target_consumption=self.ts.current("target_consumption"),
                income=self.ts.current("expected_income"),
                rent=self.ts.current("rent"),
                wealth_in_financial_assets=self.ts.current("wealth_financial_assets"),
            )
        )
        self.ts.total_target_consumption_loans.append([self.ts.current("target_consumption_loans").sum()])

        # Mortgages
        target_house_price = np.zeros(self.ts.current("n_households"))
        if len(current_sales) > 0:
            target_house_price[current_sales["buyer_id"].values] = current_sales["price_or_rent"].values
        self.ts.target_mortgage.append(
            self.functions["target_credit"].compute_target_mortgage(
                target_house_price=target_house_price,
                target_consumption=self.ts.current("target_consumption"),
                income=self.ts.current("expected_income"),
                rent=self.ts.current("rent"),
                wealth_in_financial_assets=self.ts.current("wealth_financial_assets"),
            )
        )
        self.ts.total_target_mortgage.append([self.ts.current("target_mortgage").sum()])

    def compute_interest_paid_on_deposits(
        self,
        bank_interest_rate_on_household_deposits: np.ndarray,
        bank_overdraft_rate_on_household_deposits: np.ndarray,
    ) -> np.ndarray:
        """Calculate interest paid on deposits.

        Computes interest flows based on:
        - Deposit balances
        - Interest rates
        - Overdraft conditions

        Args:
            bank_interest_rate_on_household_deposits (np.ndarray): Deposit rates
            bank_overdraft_rate_on_household_deposits (np.ndarray): Overdraft rates

        Returns:
            np.ndarray: Interest paid by household
        """
        return -bank_interest_rate_on_household_deposits[self.states["Corresponding Bank ID"]] * np.maximum(
            0.0, self.ts.current("wealth_deposits")
        ) - bank_overdraft_rate_on_household_deposits[self.states["Corresponding Bank ID"]] * np.minimum(
            0.0, self.ts.current("wealth_deposits")
        )

    def compute_interest_paid(self) -> np.ndarray:
        """Calculate total interest paid.

        Aggregates interest payments on:
        - Deposits
        - Loans
        - Credit facilities

        Returns:
            np.ndarray: Total interest paid by household
        """
        return self.ts.current("interest_paid_on_loans") + self.ts.current("interest_paid_on_deposits")

    def prepare_goods_market_clearing(
        self,
        exchange_rate_usd_to_lcu: float,
    ) -> None:
        """Prepare for goods market clearing.

        Sets up market participation through:
        - Exchange rate adjustment
        - Purchase preparation
        - Sale preparation

        Args:
            exchange_rate_usd_to_lcu (float): USD to local currency rate
        """
        # Exchange rates
        self.set_exchange_rate(exchange_rate_usd_to_lcu)

        # Prepare goods market clearing
        self.prepare_buying_goods()
        self.prepare_selling_goods()

    def prepare_buying_goods(self) -> None:
        """Prepare goods purchase decisions.

        Sets up buying based on:
        - Target consumption
        - Target investment
        - Exchange rates
        """
        self.set_goods_to_buy(
            1.0
            / self.exchange_rate_usd_to_lcu
            * (self.ts.current("target_consumption") + self.ts.current("target_investment"))
        )

    def prepare_selling_goods(self) -> None:
        """Prepare goods sale decisions.

        Sets up selling based on:
        - Available goods
        - Price levels
        """
        self.set_goods_to_sell(np.zeros(self.ts.current("n_households")))
        self.set_prices(np.zeros(self.ts.current("n_households")))

    def update_consumption_and_investment(
        self,
        tau_vat: float,
        tau_cf: float,
        add_emissions: bool = False,
        readjusted_factors: Optional[np.ndarray] = None,
        emitting_indices: Optional[np.ndarray] = None,
    ) -> None:
        """Update consumption and investment outcomes.

        Records actual:
        - Consumption spending
        - Investment spending
        - Tax payments
        - Emissions data

        Args:
            tau_vat (float): Value added tax rate
            tau_cf (float): Capital formation tax rate
            add_emissions (bool): Whether to track emissions
            readjusted_factors (Optional[np.ndarray]): Emission factors
            emitting_indices (Optional[np.ndarray]): Emitting sectors
        """
        # Total amount spent
        self.ts.amount_bought.append(self.ts.current("nominal_amount_spent_in_lcu").sum(axis=1))

        # Distribute
        consumption_by_good = np.minimum(
            self.ts.current("nominal_amount_spent_in_lcu"),
            self.ts.current("target_consumption"),
        )

        if add_emissions:
            consumption_emissions = consumption_by_good[:, emitting_indices] @ readjusted_factors
            self.ts.consumption_emissions.append(consumption_emissions)
            # coal, oil, gas, refined products consumption emissions
            disaggregated_emissions = consumption_by_good[:, emitting_indices] * readjusted_factors
            self.ts.coal_consumption_emissions.append(disaggregated_emissions[:, 0])
            self.ts.oil_consumption_emissions.append(disaggregated_emissions[:, 1])
            self.ts.gas_consumption_emissions.append(disaggregated_emissions[:, 2])
            self.ts.refined_products_consumption_emissions.append(disaggregated_emissions[:, 3])

        # Consumption
        self.ts.consumption.append(consumption_by_good.sum(axis=1))
        self.ts.total_consumption.append([(1 + tau_vat) * self.ts.current("consumption").sum()])
        self.ts.total_consumption_before_vat.append([self.ts.current("consumption").sum()])
        self.ts.industry_consumption.append(consumption_by_good.sum(axis=0))

        # Investment
        self.ts.investment.append(self.ts.current("nominal_amount_spent_in_lcu") - consumption_by_good)
        if add_emissions:
            inv = self.ts.current("nominal_amount_spent_in_lcu") - consumption_by_good
            investment_emissions = inv[:, emitting_indices] @ readjusted_factors
            self.ts.investment_emissions.append(investment_emissions)

            # coal, oil, gas, refined products investment emissions
            disaggregated_emissions = inv[:, emitting_indices] * readjusted_factors
            self.ts.coal_investment_emissions.append(disaggregated_emissions[:, 0])
            self.ts.oil_investment_emissions.append(disaggregated_emissions[:, 1])
            self.ts.gas_investment_emissions.append(disaggregated_emissions[:, 2])
            self.ts.refined_products_investment_emissions.append(disaggregated_emissions[:, 3])
        self.ts.total_investment.append([(1 + tau_cf) * self.ts.current("investment").sum()])
        self.ts.total_investment_before_vat.append([self.ts.current("investment").sum()])
        self.ts.industry_investment.append(self.ts.current("investment").sum(axis=0))

    def update_wealth(self, housing_data: pd.DataFrame, tau_cf: float) -> None:
        """Update household wealth positions.

        Updates:
        - Real asset holdings
        - Financial assets
        - Property values
        - Net wealth position

        Args:
            housing_data (pd.DataFrame): Property market data
            tau_cf (float): Capital formation tax rate
        """
        # Update real wealth
        self.ts.wealth_main_residence.append(
            self.compute_wealth_of_the_main_residence(
                housing_data=housing_data,
            )
        )
        self.ts.total_wealth_main_residence.append([self.ts.current("wealth_main_residence").sum()])
        self.ts.wealth_other_properties.append(
            self.compute_wealth_of_other_properties(
                housing_data=housing_data,
            )
        )
        self.ts.total_wealth_other_properties.append([self.ts.current("wealth_other_properties").sum()])
        self.ts.wealth_other_real_assets.append(self.compute_wealth_of_other_real_assets())
        self.ts.total_wealth_other_real_assets.append([self.ts.current("wealth_other_real_assets").sum()])
        self.ts.wealth_real_assets.append(
            self.ts.current("wealth_main_residence")
            + self.ts.current("wealth_other_properties")
            + self.ts.current("wealth_other_real_assets")
        )

        # New financial wealth
        new_wealth = np.maximum(
            0.0,
            (
                self.ts.current("income")
                - self.ts.current("rent")
                - self.ts.current("nominal_amount_spent_in_lcu").sum(axis=1)
            ),
        )
        (
            new_wealth_in_deposits,
            new_wealth_in_other_financial_assets,
        ) = self.functions["wealth"].distribute_new_wealth(
            new_wealth=new_wealth,
            model=self.states["wealth_distribution_model"],
            ts=self.ts,
        )

        # Used-up financial wealth
        used_up_wealth = -np.minimum(
            0.0,
            (
                self.ts.current("income")
                - self.ts.current("rent")
                - self.ts.current("nominal_amount_spent_in_lcu").sum(axis=1)
            ),
        )
        (
            used_up_wealth_in_deposits,
            used_up_wealth_in_other_financial_assets,
        ) = self.functions["wealth"].use_up_wealth(
            used_up_wealth=used_up_wealth,
            current_wealth_in_deposits=self.ts.current("wealth_deposits"),
            current_wealth_in_other_financial_assets=self.ts.current("wealth_other_financial_assets"),
        )

        # Update other financial assets
        self.ts.wealth_other_financial_assets.append(
            self.compute_wealth_of_other_financial_assets(
                new_wealth_in_other_financial_assets=new_wealth_in_other_financial_assets,
                used_up_wealth_in_other_financial_assets=used_up_wealth_in_other_financial_assets,
            )
        )
        self.ts.total_wealth_other_financial_assets.append([self.ts.current("wealth_other_financial_assets").sum()])

        # Update deposits
        self.ts.wealth_deposits.append(
            self.compute_wealth_in_deposits(
                new_wealth_in_deposits=new_wealth_in_deposits,
                used_up_wealth_in_deposits=used_up_wealth_in_deposits,
                tau_cf=tau_cf,
            )
        )
        self.ts.total_wealth_deposits.append([self.ts.current("wealth_deposits").sum()])

        # Compute total financial assets
        self.ts.wealth_financial_assets.append(
            self.ts.current("wealth_other_financial_assets") + self.ts.current("wealth_deposits")
        )

        # Compute total wealth
        self.ts.wealth.append(self.ts.current("wealth_real_assets") + self.ts.current("wealth_financial_assets"))

    def compute_wealth_of_the_main_residence(self, housing_data: pd.DataFrame) -> np.ndarray:
        """Calculate main residence wealth.

        Determines value of:
        - Owner-occupied housing
        - Primary residences

        Args:
            housing_data (pd.DataFrame): Property market data

        Returns:
            np.ndarray: Main residence value by household
        """
        wealth_of_the_main_residence = np.zeros(self.ts.current("n_households"))
        ind_owning_mhr = np.isin(self.states["Tenure Status of the Main Residence"], [1, 2, 4])
        wealth_of_the_main_residence[ind_owning_mhr] = housing_data.loc[
            self.states["Corresponding Inhabited House ID"][ind_owning_mhr],
            "Value",
        ].values
        return wealth_of_the_main_residence

    def compute_wealth_of_other_properties(self, housing_data: pd.DataFrame) -> np.ndarray:
        """Calculate other property wealth.

        Determines value of:
        - Investment properties
        - Rental properties
        - Secondary homes

        Args:
            housing_data (pd.DataFrame): Property market data

        Returns:
            np.ndarray: Other property value by household
        """
        wealth_of_other_properties = np.zeros(self.ts.current("n_households"))
        housing_data_not_oo = housing_data.loc[housing_data["Is Owner-Occupied"] == 0]
        housing_data_not_oo_grouped = housing_data_not_oo.groupby("Corresponding Owner Household ID")["Value"].sum()
        wealth_of_other_properties[housing_data_not_oo_grouped.index.values] = housing_data_not_oo_grouped.values
        return wealth_of_other_properties

    def compute_wealth_of_other_real_assets(self) -> np.ndarray:
        """Calculate other real asset wealth.

        Determines value of:
        - Non-property real assets
        - Physical investments
        - Durable goods

        Returns:
            np.ndarray: Other real asset value by household
        """
        return self.functions["wealth"].compute_wealth_in_other_real_assets(
            current_wealth_in_other_real_assets=self.ts.current("wealth_other_real_assets"),
            current_investment_in_other_real_assets=self.ts.current("investment").sum(axis=1),
        )

    def compute_wealth_of_other_financial_assets(
        self,
        new_wealth_in_other_financial_assets: float,
        used_up_wealth_in_other_financial_assets: float,
    ) -> np.ndarray:
        """Calculate other financial asset wealth.

        Updates financial assets based on:
        - New investments
        - Asset usage
        - Market returns

        Args:
            new_wealth_in_other_financial_assets (float): New investments
            used_up_wealth_in_other_financial_assets (float): Used assets

        Returns:
            np.ndarray: Financial asset value by household
        """
        return self.functions["wealth"].compute_wealth_in_other_financial_assets(
            current_wealth_in_other_financial_assets=self.ts.current("wealth_other_financial_assets"),
            new_wealth_in_other_financial_assets=new_wealth_in_other_financial_assets,
            used_up_wealth_in_other_financial_assets=used_up_wealth_in_other_financial_assets,
        )

    def compute_wealth_in_deposits(
        self,
        new_wealth_in_deposits: np.ndarray,
        used_up_wealth_in_deposits: np.ndarray,
        tau_cf: float,
    ) -> np.ndarray:
        """Calculate deposit wealth.

        Updates deposits based on:
        - New savings
        - Withdrawals
        - Interest earned
        - Tax effects

        Args:
            new_wealth_in_deposits (np.ndarray): New deposits
            used_up_wealth_in_deposits (np.ndarray): Used deposits
            tau_cf (float): Capital formation tax rate

        Returns:
            np.ndarray: Deposit value by household
        """
        return self.functions["wealth"].compute_wealth_in_deposits(
            current_wealth_in_deposits=self.ts.current("wealth_deposits"),
            new_wealth_in_deposits=new_wealth_in_deposits,
            used_up_wealth_in_deposits=used_up_wealth_in_deposits,
            current_interest_paid=self.ts.current("interest_paid"),
            price_paid_for_property=self.ts.current("price_paid_for_property"),
            debt_installments=self.ts.current("debt_installments"),
            new_loans=self.ts.current("received_consumption_loans") + self.ts.current("received_mortgages"),
            new_real_wealth=self.ts.current("investment").sum(axis=1),
            tau_cf=tau_cf,
        )

    def compute_debt(self) -> np.ndarray:
        """Calculate total household debt.

        Aggregates debt from:
        - Consumption loans
        - Mortgages
        - Other credit

        Returns:
            np.ndarray: Total debt by household
        """
        self.ts.total_consumption_loan_debt.append([self.ts.current("consumption_loan_debt").sum()])
        self.ts.total_mortgage_debt.append([self.ts.current("mortgage_debt").sum()])
        return self.ts.current("consumption_loan_debt") + self.ts.current("mortgage_debt")

    def compute_net_wealth(self) -> np.ndarray:
        """Calculate household net wealth.

        Determines net position from:
        - Total assets
        - Total liabilities
        - Debt obligations

        Returns:
            np.ndarray: Net wealth by household
        """
        return self.ts.current("wealth") - self.ts.current("debt")

    def handle_insolvency(self, banks: Banks, credit_market: CreditMarket) -> Tuple[float, float, float]:
        """Handle household insolvency cases.

        Processes defaults through:
        - Debt restructuring
        - Asset liquidation
        - Bank interactions

        Args:
            banks (Banks): Banking sector agent
            credit_market (CreditMarket): Credit market interface

        Returns:
            Tuple[float, float, float]: Default outcomes
        """
        return self.functions["insolvency"].handle_insolvency(
            households=self,
            banks=banks,
            credit_market=credit_market,
        )

    def save_to_h5(self, group: h5py.Group):
        """Save household data to HDF5.

        Stores:
        - Time series data
        - State variables
        - Market positions

        Args:
            group (h5py.Group): HDF5 storage group
        """
        self.ts.write_to_h5("households", group)

    def save_consumption_weights(self, group: h5py.Group):
        """Save consumption weight data.

        Stores:
        - Income-based weights
        - Industry allocations
        - Consumption patterns

        Args:
            group (h5py.Group): HDF5 storage group
        """
        group.create_dataset("household_consumption_weights_by_income", data=self.consumption_weights.T)
        group["household_consumption_weights_by_income"].attrs["columns"] = list(range(self.n_industries))

    def total_consumption(self) -> np.ndarray:
        """Get total consumption time series.

        Returns:
            np.ndarray: Aggregate consumption over time
        """
        return self.ts.get_aggregate("total_consumption")

    def consumption_loan_debt(self) -> np.ndarray:
        """Get consumption loan debt time series.

        Returns:
            np.ndarray: Aggregate consumption debt over time
        """
        return self.ts.get_aggregate("consumption_loan_debt")

    def mortgage_debt(self) -> np.ndarray:
        """Get mortgage debt time series.

        Returns:
            np.ndarray: Aggregate mortgage debt over time
        """
        return self.ts.get_aggregate("mortgage_debt")

    def consumption_emissions(self) -> np.ndarray:
        """Get consumption emissions time series.

        Returns:
            np.ndarray: Aggregate consumption emissions over time
        """
        return self.ts.get_aggregate("consumption_emissions")

    def investment_emissions(self) -> np.ndarray:
        """Get investment emissions time series.

        Returns:
            np.ndarray: Aggregate investment emissions over time
        """
        return self.ts.get_aggregate("investment_emissions")

    def disaggregated_consumption_emissions(self, input_name: str) -> np.ndarray:
        """Get disaggregated consumption emissions.

        Args:
            input_name (str): Input category name

        Returns:
            np.ndarray: Category-specific consumption emissions
        """
        return self.ts.get_aggregate(f"{input_name}_consumption_emissions")

    def disaggregated_investment_emissions(self, input_name: str) -> np.ndarray:
        """Get disaggregated investment emissions.

        Args:
            input_name (str): Input category name

        Returns:
            np.ndarray: Category-specific investment emissions
        """
        return self.ts.get_aggregate(f"{input_name}_investment_emissions")
