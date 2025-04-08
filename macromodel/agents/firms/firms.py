from copy import deepcopy
from typing import Any, Callable, Optional

import h5py
import numpy as np
import pandas as pd

import macromodel.util.get_histogram
from macro_data import SyntheticFirms
from macromodel.agents.agent import Agent
from macromodel.agents.firms.firm_ts import FirmTimeSeries
from macromodel.configurations import FirmsConfiguration
from macromodel.markets.credit_market.credit_market import CreditMarket
from macromodel.markets.goods_market.value_type import ValueType
from macromodel.util.function_mapping import functions_from_model, update_functions


class Firms(Agent):
    """A collection of producing firms in the economy.

    This class represents the productive sector of the economy, managing multiple firms
    that produce goods using labor, intermediate inputs, and capital inputs. Each firm:
    - Makes production decisions based on expected demand and capacity
    - Sets prices based on costs and market conditions
    - Hires workers and sets wages
    - Manages inventory and input stocks
    - Makes investment decisions
    - Handles financial decisions (borrowing, etc.)
    - Can become insolvent

    The firms operate in discrete time steps, with each period involving:
    1. Planning (production targets, input needs, hiring)
    2. Market participation (labor, credit, goods markets)
    3. Production and sales
    4. Financial settlement (wages, loans, taxes)

    Attributes:
        country_name (str): Country the firms operate in
        all_country_names (list[str]): All countries in simulation
        n_industries (int): Number of industry sectors
        functions (dict[str, Callable]): Production and decision functions
        ts (FirmTimeSeries): Time series data for firms
        states (dict[str, np.ndarray]): Current state variables
        intermediate_inputs_productivity_matrix (np.ndarray): Input-output coefficients
        capital_inputs_productivity_matrix (np.ndarray): Capital productivity coefficients
        capital_inputs_depreciation_matrix (np.ndarray): Capital depreciation rates
        goods_criticality_matrix (np.ndarray): Critical input requirements
        intermediate_inputs_utilisation_rate (float): Input capacity utilization
        capital_inputs_utilisation_rate (float): Capital capacity utilization
        capital_inputs_delay (np.ndarray): Investment implementation lags
        depreciation_rates (np.ndarray): Asset depreciation rates
        average_initial_price (np.ndarray): Initial price levels
        configuration (FirmsConfiguration): Model parameters
        industries (list[str]): Industry sector names
    """

    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        functions: dict[str, Callable],
        ts: FirmTimeSeries,
        states: dict[str, np.ndarray],
        intermediate_inputs_productivity_matrix: np.ndarray,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        goods_criticality_matrix: np.ndarray,
        intermediate_inputs_utilisation_rate: float,
        capital_inputs_utilisation_rate: float,
        depreciation_rates: np.ndarray,
        capital_inputs_delay: np.ndarray,
        average_initial_price: np.ndarray,
        configuration: FirmsConfiguration,
        industries: list[str],
    ):
        """Initialize the firms sector.

        Args:
            country_name (str): Country identifier
            all_country_names (list[str]): All countries in simulation
            n_industries (int): Number of industry sectors
            functions (dict[str, Callable]): Production and decision functions
            ts (FirmTimeSeries): Time series container
            states (dict[str, np.ndarray]): Initial state variables
            intermediate_inputs_productivity_matrix (np.ndarray): Input-output coefficients
            capital_inputs_productivity_matrix (np.ndarray): Capital productivity coefficients
            capital_inputs_depreciation_matrix (np.ndarray): Capital depreciation rates
            goods_criticality_matrix (np.ndarray): Critical input requirements
            intermediate_inputs_utilisation_rate (float): Input capacity utilization
            capital_inputs_utilisation_rate (float): Capital capacity utilization
            depreciation_rates (np.ndarray): Asset depreciation rates
            capital_inputs_delay (np.ndarray): Investment implementation lags
            average_initial_price (np.ndarray): Initial price levels
            configuration (FirmsConfiguration): Model parameters
            industries (list[str]): Industry sector names
        """
        n_transactors = ts.current("n_firms")
        super().__init__(
            country_name,
            all_country_names,
            n_industries,
            n_transactors,
            n_transactors,
            ts,
            states,
            transactor_settings={
                "Buyer Value Type": ValueType.REAL,
                "Seller Value Type": ValueType.REAL,
                "Buyer Priority": 1,
                "Seller Priority": 1,
            },
        )

        self.functions: dict[str, Any] = functions
        self.intermediate_inputs_productivity_matrix = intermediate_inputs_productivity_matrix
        self.capital_inputs_productivity_matrix = capital_inputs_productivity_matrix
        self.capital_inputs_depreciation_matrix = capital_inputs_depreciation_matrix
        self.goods_criticality_matrix = goods_criticality_matrix
        self.intermediate_inputs_utilisation_rate = intermediate_inputs_utilisation_rate
        self.capital_inputs_utilisation_rate = capital_inputs_utilisation_rate
        self.capital_inputs_delay = capital_inputs_delay
        self.depreciation_rates = depreciation_rates

        self.average_initial_price = average_initial_price

        self.configuration = configuration

        self.industries = industries

    @classmethod
    def from_pickled_agent(
        cls,
        synthetic_firms: SyntheticFirms,
        configuration: FirmsConfiguration,
        country_name: str,
        all_country_names: list[str],
        goods_criticality_matrix: pd.DataFrame | np.ndarray,
        average_initial_price: np.ndarray,
        industries: list[str],
        add_emissions: bool = False,
    ):
        """Create a Firms instance from pickled synthetic data.

        Factory method that constructs a Firms object from serialized synthetic data,
        configuration parameters, and initial economic conditions.

        Args:
            synthetic_firms (SyntheticFirms): Synthetic firm data
            configuration (FirmsConfiguration): Model configuration parameters
            country_name (str): Country identifier
            all_country_names (list[str]): All countries in simulation
            goods_criticality_matrix (pd.DataFrame | np.ndarray): Input criticality coefficients
            average_initial_price (np.ndarray): Initial price levels
            industries (list[str]): Industry sector names
            add_emissions (bool, optional): Whether to track emissions. Defaults to False.

        Returns:
            Firms: Initialized Firms instance
        """
        functions = functions_from_model(model=configuration.functions, loc="macromodel.agents.firms")

        intermediate_inputs_productivity_matrix = synthetic_firms.intermediate_inputs_productivity_matrix
        capital_inputs_productivity_matrix = synthetic_firms.capital_inputs_productivity_matrix
        capital_inputs_depreciation_matrix = synthetic_firms.capital_inputs_depreciation_matrix
        if isinstance(goods_criticality_matrix, pd.DataFrame):
            goods_criticality_matrix = goods_criticality_matrix.values

        corr_employees = synthetic_firms.firm_data["Employees ID"]

        corr_employees = [[int(x) for x in sublist if not pd.isna(x)] for sublist in corr_employees]

        data = synthetic_firms.firm_data.drop(columns=["Employees ID"]).astype(float).rename_axis("Firm ID")

        if add_emissions:
            inputs_emissions = synthetic_firms.firm_data["Input Emissions"].values
            capital_emissions = synthetic_firms.firm_data["Capital Emissions"].values
            input_dict: dict = {
                f"{key}_inputs_emissions": synthetic_firms.firm_data[f"{emitting_industry} Input Emissions"]
                for key, emitting_industry in zip(
                    ["coal", "oil", "gas", "refined_products"], ["Coal", "Oil", "Gas", "Refined Products"]
                )
            }
            capital_dict = {
                f"{key}_capital_emissions": synthetic_firms.firm_data[f"{emitting_industry} Capital Emissions"]
                for key, emitting_industry in zip(
                    ["coal", "oil", "gas", "refined_products"], ["Coal", "Oil", "Gas", "Refined Products"]
                )
            }

        else:
            inputs_emissions = None
            capital_emissions = None
            input_dict = {}
            capital_dict = {}

        ts = FirmTimeSeries.from_data(
            data=data,
            intermediate_inputs_stock=synthetic_firms.intermediate_inputs_stock,
            used_intermediate_inputs=synthetic_firms.used_intermediate_inputs,
            capital_inputs_stock=synthetic_firms.capital_inputs_stock,
            used_capital_inputs=synthetic_firms.used_capital_inputs,
            initial_good_prices=average_initial_price,
            n_industries=len(synthetic_firms.industries),
            calculate_hill_exponent=configuration.calculate_hill_exponent,
            inputs_emissions=inputs_emissions,
            capital_emissions=capital_emissions,
            **input_dict,
            **capital_dict,
        )

        states: dict[str, Any] = {}

        for state_name in [
            "Industry",
            "Corresponding Bank ID",
        ]:
            if state_name not in data.columns:
                raise ValueError("Missing " + state_name + " from the data for initialising firms.")
            states[state_name] = data[state_name].fillna(-1).values.astype(int)

        states["Employments"] = corr_employees
        states["is_insolvent"] = np.full(data.shape[0], False)
        states["Excess Demand"] = np.zeros(data.shape[0])

        states["Labour Productivity by Industry"] = synthetic_firms.labour_productivity_by_industry

        return cls(
            country_name,
            all_country_names,
            len(synthetic_firms.industries),
            functions,
            ts,
            states,
            intermediate_inputs_productivity_matrix,
            capital_inputs_productivity_matrix,
            capital_inputs_depreciation_matrix,
            goods_criticality_matrix,
            configuration.parameters.intermediate_inputs_utilisation_rate,
            configuration.parameters.capital_inputs_utilisation_rate,
            np.array(configuration.parameters.depreciation_rates),
            np.array(configuration.parameters.capital_inputs_delay),
            average_initial_price,
            configuration=configuration,
            industries=industries,
        )

    @property
    def industries_dataframe(self) -> pd.DataFrame:
        """Get a DataFrame mapping firms to their industries.

        Returns a DataFrame with firm indices and their corresponding industry names.

        Returns:
            pd.DataFrame: DataFrame with 'Firm_ID' as the index and 'Industry' as the column.
        """
        # Get the industry indices of the firms
        industry_indices = self.states["Industry"]

        # Map industry indices to industry names
        industry_names = [self.industries[i] for i in industry_indices]

        # Assuming firms are indexed from 0 to N-1
        firm_indices = np.arange(len(industry_indices))

        # Create the DataFrame
        df = pd.DataFrame({"Industry": industry_names}, index=firm_indices)
        df.index.name = "Firm_ID"

        return df

    def reset(self, configuration: FirmsConfiguration) -> None:
        """Reset the firms to initial state with new configuration.

        Resets all time series and state variables to initial values,
        updates function configurations, and recalculates initial stocks
        based on the new configuration parameters.

        Args:
            configuration (FirmsConfiguration): New model configuration
        """
        self.gen_reset()
        update_functions(model=configuration.functions, loc="macromodel.agents.firms", functions=self.functions)

        current_inv = (
            self.ts.current("production")
            * configuration.functions.target_production.parameters["target_inventory_to_production_fraction"]
        )

        industries = self.states["Industry"]
        initial_good_prices = self.average_initial_price

        inter_inputs_stock = (
            1.0
            / configuration.parameters.intermediate_inputs_utilisation_rate
            * np.divide(
                self.ts.current("production"),
                self.intermediate_inputs_productivity_matrix[:, industries],
                out=np.zeros(self.intermediate_inputs_productivity_matrix[:, industries].shape),
                where=self.intermediate_inputs_productivity_matrix[:, industries] != 0.0,
            ).T
        )

        cap_inputs_stock = (
            1.0
            / configuration.parameters.capital_inputs_utilisation_rate
            * np.divide(
                self.ts.current("production"),
                self.capital_inputs_productivity_matrix[:, industries],
                out=np.zeros(self.capital_inputs_productivity_matrix[:, industries].shape),
                where=self.capital_inputs_productivity_matrix[:, industries] != 0.0,
            ).T
        )

        self.ts.reset_values(
            inventory=current_inv,
            initial_good_prices=initial_good_prices,
            intermediate_inputs_stock=inter_inputs_stock,
            capital_inputs_stock=cap_inputs_stock,
        )

        self.configuration = deepcopy(configuration)

    def update_number_of_firms(self) -> None:
        """Update the time series tracking number of firms.

        Records the current number of total firms and firms by industry
        in the time series data.
        """
        self.ts.n_firms.append(self.ts.current("n_firms"))
        self.ts.n_firms_by_industry.append(self.ts.current("n_firms_by_industry"))

    def set_estimates(
        self,
        current_estimated_growth: float,
        previous_average_good_prices: np.ndarray,
    ) -> None:
        """Set growth and demand estimates for firms.

        Updates time series with estimated growth rates by firm and
        estimated demand based on current economic conditions.

        Args:
            current_estimated_growth (float): Overall estimated growth rate
            previous_average_good_prices (np.ndarray): Previous period prices
        """
        self.ts.estimated_growth_by_firm.append(
            self.compute_estimated_growth_by_firm(previous_average_good_prices=previous_average_good_prices)
        )
        self.ts.estimated_demand.append(
            self.compute_estimated_demand(current_estimated_growth=current_estimated_growth)
        )

    def compute_estimated_growth_by_firm(
        self,
        previous_average_good_prices: np.ndarray,
        min_growth: float = -0.2,
        max_growth: float = 0.2,
    ) -> np.ndarray:
        """Calculate expected growth rates for each firm.

        Estimates firm-specific growth rates based on:
        - Price changes
        - Supply and demand conditions
        - Industry dynamics
        Growth rates are bounded between min_growth and max_growth.

        Args:
            previous_average_good_prices (np.ndarray): Previous period prices
            min_growth (float, optional): Minimum growth rate. Defaults to -0.2.
            max_growth (float, optional): Maximum growth rate. Defaults to 0.2.

        Returns:
            np.ndarray: Expected growth rate for each firm
        """
        if len(self.ts.historic("inventory")) == 1:
            prev_supply = self.ts.current("production") + self.ts.current("inventory")
        else:
            prev_supply = self.ts.current("production") + self.ts.prev("inventory")
        growth = self.functions["growth_estimator"].compute_growth(
            prev_average_good_prices=previous_average_good_prices,
            prev_firm_prices=self.ts.current("price"),
            prev_supply=prev_supply,
            prev_demand=self.ts.current("demand"),
            current_firm_sectors=self.states["Industry"],
        )
        return np.maximum(min_growth, np.minimum(max_growth, growth))

    def compute_estimated_demand(
        self,
        current_estimated_growth: float,
    ) -> np.ndarray:
        """Estimate future demand for each firm's output.

        Projects demand based on:
        - Previous period demand
        - Overall economic growth
        - Firm-specific growth expectations

        Args:
            current_estimated_growth (float): Overall estimated growth rate

        Returns:
            np.ndarray: Estimated demand for each firm
        """
        return self.functions["demand_estimator"].compute_estimated_demand(
            previous_demand=self.ts.current("demand"),
            current_estimated_growth=current_estimated_growth,
            estimated_growth_by_firm=self.ts.current("estimated_growth_by_firm"),
        )

    def set_targets(
        self,
        bank_overdraft_rate_on_firm_deposits: np.ndarray,
    ) -> None:
        """Set production and input targets for firms.

        Updates time series with:
        - Input constraints (intermediate and capital)
        - Target production levels
        - Desired labor inputs
        - Target input purchases

        Args:
            bank_overdraft_rate_on_firm_deposits (np.ndarray): Overdraft interest rates
        """
        self.ts.limiting_intermediate_inputs.append(
            self.functions["production"].compute_limiting_intermediate_inputs_stock(
                intermediate_inputs_productivity_matrix=self.intermediate_inputs_productivity_matrix[
                    :, self.states["Industry"]
                ].T,
                intermediate_inputs_stock=self.ts.current("intermediate_inputs_stock"),
                intermediate_inputs_utilisation_rate=self.intermediate_inputs_utilisation_rate,
                goods_criticality_matrix=self.goods_criticality_matrix,
            )
        )
        self.ts.limiting_capital_inputs.append(
            self.functions["production"].compute_limiting_capital_inputs_stock(
                capital_inputs_productivity_matrix=self.capital_inputs_productivity_matrix[
                    :, self.states["Industry"]
                ].T,
                capital_inputs_stock=self.ts.current("capital_inputs_stock"),
                capital_inputs_utilisation_rate=self.capital_inputs_utilisation_rate,
                goods_criticality_matrix=self.goods_criticality_matrix,
            )
        )
        self.ts.target_production.append(
            self.compute_target_production(
                bank_overdraft_rate_on_firm_deposits=bank_overdraft_rate_on_firm_deposits,
            )
        )
        self.ts.desired_labour_inputs.append(self.compute_desired_labour_inputs())
        self.ts.target_intermediate_inputs_production.append(self.compute_target_intermediate_inputs_production())
        self.ts.target_capital_inputs_production.append(self.compute_target_capital_inputs_production())

    def compute_estimated_profits(self, estimated_growth: float, estimated_inflation: float) -> np.ndarray:
        """Estimate future profits for each firm.

        Uses the profit estimator function to project future profits based on
        current profits and expected economic conditions.

        Args:
            estimated_growth (float): Expected real growth rate
            estimated_inflation (float): Expected inflation rate

        Returns:
            np.ndarray: Estimated profits for each firm
        """
        return self.functions["profit_estimator"].compute_estimated_profits(
            current_profits=self.ts.current("profits"),
            estimated_growth=estimated_growth,
            estimated_inflation=estimated_inflation,
        )

    def compute_target_production(
        self,
        bank_overdraft_rate_on_firm_deposits: np.ndarray,
    ) -> np.ndarray:
        """Calculate production targets for each firm.

        Sets target production levels based on:
        - Estimated demand
        - Current inventory levels
        - Input constraints (labor, intermediate, capital)
        - Financial constraints (equity, debt capacity)

        Args:
            bank_overdraft_rate_on_firm_deposits (np.ndarray): Overdraft interest rates

        Returns:
            np.ndarray: Target production quantities for each firm
        """
        return self.functions["target_production"].compute_target_production(
            current_estimated_demand=self.ts.current("estimated_demand"),
            initial_inventory=self.ts.initial("inventory"),
            previous_inventory=self.ts.current("inventory"),
            previous_production=self.ts.current("production"),
            current_target_production=self.ts.current("target_production"),
            current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            current_limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
            current_firm_equity=self.ts.current("equity"),
            current_firm_debt=self.ts.current("debt"),
            previous_loans_applied_for=self.ts.current("target_short_term_credit")
            + self.ts.current("target_long_term_credit"),
            current_firm_deposits=self.ts.current("deposits"),
            interest_on_overdraft_rates=-bank_overdraft_rate_on_firm_deposits[self.states["Corresponding Bank ID"]]
            * np.minimum(0.0, self.ts.current("deposits")),
            interest_paid_on_loans=self.ts.current("interest_paid_on_loans"),
        )

    def compute_target_intermediate_inputs_production(self) -> np.ndarray:
        """Calculate target intermediate input production levels.

        Determines constrained intermediate input targets based on:
        - Previous production
        - Target production
        - Input constraints (labor, intermediate, capital)
        - Financial constraints (equity, debt)

        Returns:
            np.ndarray: Target intermediate input production for each firm
        """

        target_intermediate_inputs = self.functions[
            "target_production"
        ].compute_constrained_intermediate_inputs_target_production(
            previous_production=self.ts.current("production"),
            current_target_production=self.ts.current("target_production"),
            current_limiting_labour_inputs=self.ts.current("labour_inputs"),
            current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            current_limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
            current_firm_equity=self.ts.current("equity"),
            current_firm_debt=self.ts.current("debt"),
            previous_loans_applied_for=self.ts.current("target_short_term_credit")
            + self.ts.current("target_long_term_credit"),
        )

        target_intermediate_inputs = fillna(target_intermediate_inputs)

        return target_intermediate_inputs

    def compute_target_capital_inputs_production(self) -> np.ndarray:
        """Calculate target capital input production levels.

        Determines constrained capital input targets based on:
        - Previous production
        - Target production
        - Input constraints (labor, intermediate, capital)
        - Financial constraints (equity, debt)

        Returns:
            np.ndarray: Target capital input production for each firm
        """

        target_capital_inputs = self.functions[
            "target_production"
        ].compute_constrained_capital_inputs_target_production(
            previous_production=self.ts.current("production"),
            current_target_production=self.ts.current("target_production"),
            current_limiting_labour_inputs=self.ts.current("labour_inputs"),
            current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            current_limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
            current_firm_equity=self.ts.current("equity"),
            current_firm_debt=self.ts.current("debt"),
            previous_loans_applied_for=self.ts.current("target_short_term_credit")
            + self.ts.current("target_long_term_credit"),
        )

        target_capital_inputs = fillna(target_capital_inputs)

        return target_capital_inputs

    def compute_desired_labour_inputs(self) -> np.ndarray:
        """Calculate desired labor inputs for production.

        Determines optimal labor requirements based on:
        - Target production levels
        - Intermediate input constraints
        - Capital input constraints

        Returns:
            np.ndarray: Desired labor inputs for each firm
        """
        return self.functions["desired_labour"].compute_desired_labour(
            current_target_production=self.ts.current("target_production"),
            current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            current_limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
        )

    def compute_labour_inputs(self, corresponding_firm: np.ndarray, current_labour_inputs: np.ndarray) -> np.ndarray:
        """Calculate effective labor inputs for each firm.

        Computes actual labor inputs based on:
        - Employee assignments
        - Labor productivity factors
        - Industry-specific productivity

        Args:
            corresponding_firm (np.ndarray): Mapping of employees to firms
            current_labour_inputs (np.ndarray): Raw labor inputs per employee

        Returns:
            np.ndarray: Effective labor inputs per firm
        """
        labour_inputs_from_employees = np.bincount(
            corresponding_firm[corresponding_firm >= 0],
            weights=current_labour_inputs[corresponding_firm >= 0],
            minlength=self.ts.current("n_firms"),
        )
        industry_labour_productivity_by_firm = self.states["Labour Productivity by Industry"][self.states["Industry"]]

        # Compute labour productivity
        self.ts.labour_productivity_factor.append(
            self.functions["labour_productivity"].compute_labour_productivity_factor(
                current_target_production=self.ts.current("target_production"),
                current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
                current_limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
                labour_inputs_from_employees=labour_inputs_from_employees,
                industry_labour_productivity_by_firm=industry_labour_productivity_by_firm,
            )
        )
        self.ts.labour_productivity.append(
            self.ts.current("labour_productivity_factor") * industry_labour_productivity_by_firm
        )

        # Compute labour inputs
        self.ts.labour_inputs.append(self.ts.current("labour_productivity") * labour_inputs_from_employees)
        self.ts.normalised_labour_inputs.append(industry_labour_productivity_by_firm * labour_inputs_from_employees)

        return labour_inputs_from_employees

    def compute_n_employees(self, corresponding_firm: np.ndarray) -> np.ndarray:
        """Count number of employees per firm.

        Args:
            corresponding_firm (np.ndarray): Mapping of employees to firms

        Returns:
            np.ndarray: Number of employees for each firm
        """
        return np.bincount(
            corresponding_firm[corresponding_firm >= 0],
            minlength=self.ts.current("n_firms"),
        )

    def compute_production(self) -> np.ndarray:
        """Calculate actual production quantities.

        Determines production based on:
        - Target production levels
        - Labor input constraints
        - Intermediate input constraints
        - Capital input constraints

        Returns:
            np.ndarray: Actual production quantity for each firm
        """
        return self.functions["production"].compute_production(
            desired_production=self.ts.current("target_production"),
            current_labour_inputs=self.ts.current("labour_inputs"),
            current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            current_limiting_capital_inputs=self.ts.current("limiting_intermediate_inputs"),
        )

    def compute_total_sales(self) -> np.ndarray:
        """Calculate total sales revenue net of production taxes.

        Returns:
            np.ndarray: Net sales revenue for each firm
        """
        return self.ts.current("price") * self.ts.current("production") - self.ts.current("taxes_paid_on_production")

    def compute_wages_markup(self) -> np.ndarray:
        """Calculate wage markup based on labor market tightness.

        Computes markup factor based on:
        - Historical desired labor inputs
        - Historical realized labor inputs

        Returns:
            np.ndarray: Wage markup factor for each firm
        """
        return self.functions["wage_setter"].compute_wage_tightness_markup(
            historic_desired_labour_inputs=self.ts.historic("desired_labour_inputs"),
            historic_realised_labour_inputs=self.ts.historic("labour_inputs"),
        )

    def compute_offered_wage_function(
        self,
        corresponding_firm: np.ndarray,
        current_individual_labour_inputs: np.ndarray,
        previous_employee_income: np.ndarray,
        unemployment_benefits_by_individual: float,
        income_taxes: float,
        employee_social_insurance_tax: float,
        employer_social_insurance_tax: float,
    ) -> Callable[[int, float | np.ndarray], float | np.ndarray]:
        """Create function that computes offered wages.

        Returns a function that calculates wage offers based on:
        - Employee productivity
        - Previous wages
        - Labor market conditions
        - Tax rates
        - Unemployment benefits

        Args:
            corresponding_firm (np.ndarray): Mapping of employees to firms
            current_individual_labour_inputs (np.ndarray): Current labor inputs
            previous_employee_income (np.ndarray): Previous wages
            unemployment_benefits_by_individual (float): Unemployment benefit rate
            income_taxes (float): Income tax rate
            employee_social_insurance_tax (float): Employee SI tax rate
            employer_social_insurance_tax (float): Employer SI tax rate

        Returns:
            np.ndarray: Wage offers for each firm
        """
        return self.functions["wage_setter"].get_offered_wage_given_labour_inputs_function(
            corresponding_firm=corresponding_firm,
            current_individual_labour_inputs=current_individual_labour_inputs,
            previous_employee_income=previous_employee_income,
            current_target_production=self.ts.current("target_production"),
            current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            current_limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
            industry_labour_productivity_by_firm=self.states["Labour Productivity by Industry"][
                self.states["Industry"]
            ],
            initial_wage_per_capita=self.ts.initial("real_wage_per_capita"),
            current_wage_per_capita=self.ts.current("real_wage_per_capita"),
            current_labour_productivity_factor=self.ts.current("labour_productivity_factor"),
            prev_labour_productivity_factor=self.ts.prev("labour_productivity_factor"),
            current_wage_tightness_markup=self.ts.current("wage_tightness_markup"),
            income_taxes=income_taxes,
            employee_social_insurance_tax=employee_social_insurance_tax,
            employer_social_insurance_tax=employer_social_insurance_tax,
            unemployment_benefits_by_individual=unemployment_benefits_by_individual,
        )

    def set_employee_income(
        self,
        corresponding_firm: np.ndarray,
        current_individual_labour_inputs: np.ndarray,
        current_individual_stating_new_job: np.ndarray,
        current_employee_income: np.ndarray,
        current_individual_offered_wage: np.ndarray,
        labour_inputs_from_employees: np.ndarray,
        estimated_ppi_inflation: float,
        income_taxes: float,
        employee_social_insurance_tax: float,
        employer_social_insurance_tax: float,
    ) -> np.ndarray:
        """Set employee wages based on offers and market conditions.

        Updates wages considering:
        - New vs existing employees
        - Offered wages
        - Expected inflation
        - Tax rates
        - Labor productivity

        Args:
            corresponding_firm (np.ndarray): Mapping of employees to firms
            current_individual_labour_inputs (np.ndarray): Labor inputs
            current_individual_stating_new_job (np.ndarray): New job indicators
            current_employee_income (np.ndarray): Current wages
            current_individual_offered_wage (np.ndarray): Wage offers
            labour_inputs_from_employees (np.ndarray): Labor inputs by firm
            estimated_ppi_inflation (float): Expected inflation
            income_taxes (float): Income tax rate
            employee_social_insurance_tax (float): Employee SI tax rate
            employer_social_insurance_tax (float): Employer SI tax rate

        Returns:
            np.ndarray: Updated employee wages
        """
        return self.functions["wage_setter"].set_employee_income(
            corresponding_firm=corresponding_firm,
            current_individual_labour_inputs=current_individual_labour_inputs,
            current_individual_stating_new_job=current_individual_stating_new_job,
            current_employee_income=current_employee_income,
            current_individual_offered_wage=current_individual_offered_wage,
            current_target_production=self.ts.current("target_production"),
            current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            current_limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
            labour_inputs_from_employees=labour_inputs_from_employees,
            industry_labour_productivity_by_firm=self.states["Labour Productivity by Industry"][
                self.states["Industry"]
            ],
            initial_wage_per_capita=self.ts.initial("real_wage_per_capita"),
            current_wage_per_capita=self.ts.current("real_wage_per_capita"),
            current_labour_productivity_factor=self.ts.current("labour_productivity_factor"),
            prev_labour_productivity_factor=self.ts.prev("labour_productivity_factor"),
            current_wage_tightness_markup=self.ts.current("wage_tightness_markup"),
            estimated_ppi_inflation=estimated_ppi_inflation,
            income_taxes=income_taxes,
            employee_social_insurance_tax=employee_social_insurance_tax,
            employer_social_insurance_tax=employer_social_insurance_tax,
        )

    def update_total_wages_paid(
        self,
        corresponding_firm: np.ndarray,
        individual_wages: np.ndarray,
        income_taxes: float,
        employee_social_insurance_tax: float,
        employer_social_insurance_tax: float,
        cpi: float,
    ) -> None:
        """Update total wage payments including taxes and adjustments.

        Calculates total wage costs including:
        - Base wages
        - Employer social insurance contributions
        - Tax adjustments
        - Real wage per capita

        Args:
            corresponding_firm (np.ndarray): Mapping of employees to firms
            individual_wages (np.ndarray): Individual wage rates
            income_taxes (float): Income tax rate
            employee_social_insurance_tax (float): Employee SI tax rate
            employer_social_insurance_tax (float): Employer SI tax rate
            cpi (float): Consumer price index
        """
        real_wages = np.bincount(
            corresponding_firm[corresponding_firm >= 0],
            weights=individual_wages[corresponding_firm >= 0],
            minlength=self.ts.current("n_firms"),
        )
        self.ts.total_wage.append(
            cpi
            * (
                (1.0 + employer_social_insurance_tax)
                / (1 - employee_social_insurance_tax - income_taxes * (1 - employee_social_insurance_tax))
                * real_wages
            )
        )
        self.ts.real_wage_per_capita.append(
            self.ts.current("total_wage") / cpi / self.ts.current("number_of_employees")
        )

    def compute_price(
        self,
        current_estimated_ppi_inflation: np.ndarray,
        previous_average_good_prices: np.ndarray,
        ppi_during: np.ndarray,
    ) -> np.ndarray:
        """Set prices for each firm's output.

        Determines prices based on:
        - Previous prices
        - Expected inflation
        - Market conditions (excess demand)
        - Inventory levels
        - Production costs
        - Industry dynamics

        Args:
            current_estimated_ppi_inflation (np.ndarray): Expected PPI inflation
            previous_average_good_prices (np.ndarray): Previous period prices
            ppi_during (np.ndarray): Producer price indices

        Returns:
            np.ndarray: New prices for each firm
        """
        return self.functions["prices"].compute_price(
            prev_prices=self.ts.current("price"),
            current_estimated_ppi_inflation=current_estimated_ppi_inflation,
            excess_demand=self.states["Excess Demand"],
            inventories=self.ts.current("inventory"),
            production=self.ts.current("production"),
            prev_average_good_prices=previous_average_good_prices,
            prev_firm_prices=self.ts.current("price"),
            prev_supply=(
                self.ts.current("production") + self.ts.current("inventory")
                if len(self.ts.historic("price")) == 1
                else self.ts.prev("production") + self.ts.current("inventory")
            ),
            prev_demand=self.ts.current("demand"),
            current_firm_sectors=self.states["Industry"],
            curr_unit_costs=self.ts.current("unit_costs"),
            prev_unit_costs=(
                self.ts.current("unit_costs") if len(self.ts.historic("price")) == 1 else self.ts.prev("unit_costs")
            ),
            ppi_during=ppi_during,
            current_time=len(self.ts.historic("price")),
        )

    def compute_unconstrained_demand_for_intermediate_inputs(
        self,
    ) -> np.ndarray:
        """Calculate unconstrained demand for intermediate inputs.

        Determines optimal intermediate input requirements without
        considering financial or supply constraints, based on:
        - Target production
        - Input-output coefficients
        - Current stocks
        - Production history

        Returns:
            np.ndarray: Unconstrained intermediate input demand for each firm
        """
        return self.functions["target_intermediate_inputs"].compute_unconstrained_target_intermediate_inputs(
            current_target_production=self.ts.current("target_intermediate_inputs_production"),
            intermediate_inputs_productivity_matrix=self.intermediate_inputs_productivity_matrix[
                :, self.states["Industry"]
            ].T,
            prev_intermediate_inputs_stock=self.ts.current("intermediate_inputs_stock"),
            initial_intermediate_inputs_stock=self.ts.initial("intermediate_inputs_stock"),
            prev_production=self.ts.current("production"),
            initial_production=self.ts.initial("production"),
        )

    def compute_unconstrained_demand_for_intermediate_inputs_value(self, current_good_prices: np.ndarray) -> np.ndarray:
        """Calculate value of unconstrained intermediate input demand.

        Computes the monetary value of desired intermediate inputs
        using current market prices.

        Args:
            current_good_prices (np.ndarray): Current prices for inputs

        Returns:
            np.ndarray: Value of unconstrained intermediate input demand for each firm
        """
        return np.matmul(
            self.ts.current("unconstrained_target_intermediate_inputs"),
            current_good_prices,
        )

    def compute_unconstrained_demand_for_capital_inputs(self) -> np.ndarray:
        """Calculate unconstrained demand for capital inputs.

        Determines optimal capital input requirements without
        considering financial or supply constraints, based on:
        - Target production
        - Capital productivity coefficients
        - Current capital stock
        - Production history

        Returns:
            np.ndarray: Unconstrained capital input demand for each firm
        """
        return self.functions["target_capital_inputs"].compute_unconstrained_target_capital_inputs(
            current_target_production=self.ts.current("target_capital_inputs_production"),
            capital_inputs_depreciation_matrix=self.capital_inputs_depreciation_matrix[:, self.states["Industry"]].T,
            prev_capital_inputs_stock=self.ts.current("capital_inputs_stock"),
            initial_capital_inputs_stock=self.ts.initial("capital_inputs_stock"),
            prev_production=self.ts.current("production"),
            initial_production=self.ts.initial("production"),
        )

    def compute_unconstrained_demand_for_capital_inputs_value(self, current_good_prices: np.ndarray) -> np.ndarray:
        return np.matmul(
            self.ts.current("unconstrained_target_capital_inputs"),
            current_good_prices,
        )

    def compute_target_credit(self, estimated_growth: float, estimated_inflation: float) -> None:
        """Calculate target borrowing levels for each firm.

        Determines desired short-term and long-term credit based on:
        - Expected growth and inflation
        - Projected cash flows
        - Input purchase needs
        - Investment plans

        Args:
            estimated_growth (float): Expected real growth rate
            estimated_inflation (float): Expected inflation rate
        """
        estimated_corporate_taxes = (
            (1 + estimated_growth) * (1 + estimated_inflation) * self.ts.current("corporate_taxes_paid")
        )
        estimated_change_in_deposits = (
            self.ts.current("price") * self.ts.current("production")
            - self.ts.current("total_wage")
            - self.ts.current("labour_costs")
            - self.ts.current("taxes_paid_on_production")
            - estimated_corporate_taxes
            - self.ts.current("interest_paid")
            - self.ts.current("debt_installments")
        )
        estimated_deposits = self.ts.current("deposits") + estimated_change_in_deposits
        target_short_term_credit, target_long_term_credit = self.functions["target_credit"].compute_target_credit(
            estimated_deposits=estimated_deposits,
            unconstrained_target_intermediate_inputs_costs=self.ts.current(
                "unconstrained_target_intermediate_inputs_costs"
            ),
            unconstrained_target_capital_inputs_costs=self.ts.current("unconstrained_target_capital_inputs_costs"),
        )
        self.ts.target_short_term_credit.append(target_short_term_credit)
        self.ts.total_target_short_term_credit.append([target_short_term_credit.sum()])
        self.ts.target_long_term_credit.append(target_long_term_credit)
        self.ts.total_target_long_term_credit.append([target_long_term_credit.sum()])

    def compute_debt(self) -> np.ndarray:
        return self.ts.current("short_term_loan_debt") + self.ts.current("long_term_loan_debt")

    def compute_interest_paid_on_deposits(
        self,
        bank_interest_rate_on_firm_deposits: np.ndarray,
        bank_overdraft_rate_on_firm_deposits: np.ndarray,
    ) -> np.ndarray:
        return -(
            bank_interest_rate_on_firm_deposits[self.states["Corresponding Bank ID"]]
            * np.maximum(0.0, self.ts.current("deposits"))
            + bank_overdraft_rate_on_firm_deposits[self.states["Corresponding Bank ID"]]
            * np.minimum(0.0, self.ts.current("deposits"))
        )

    def compute_interest_paid(self) -> np.ndarray:
        """Calculate total interest payments.

        Sums interest paid on:
        - Loans
        - Deposits/overdrafts

        Returns:
            np.ndarray: Total interest paid by each firm
        """
        return self.ts.current("interest_paid_on_loans") + self.ts.current("interest_paid_on_deposits")

    def compute_offered_price(self) -> np.ndarray:
        """Calculate offered prices by industry.

        Computes weighted average prices based on:
        - Current prices
        - Production quantities
        - Inventory levels

        Returns:
            np.ndarray: Average offered price by industry
        """
        nom = np.bincount(
            self.states["Industry"],
            weights=self.ts.current("price_in_usd") * (self.ts.current("production") + self.ts.current("inventory")),
            minlength=self.n_industries,
        )
        real = np.bincount(
            self.states["Industry"],
            weights=self.ts.current("production") + self.ts.current("inventory"),
            minlength=self.n_industries,
        )
        avg_price = np.divide(nom, real, out=np.zeros(nom.shape), where=real != 0.0)
        avg_price[avg_price == 0.0] = self.ts.current("price_offered")[avg_price == 0.0]
        assert np.all(avg_price > 0.0)
        return avg_price

    def compute_maximum_excess_demand(self) -> np.ndarray:
        """Calculate maximum potential excess demand.

        Determines maximum additional demand that could be met based on:
        - Current production
        - Target production
        - Input constraints

        Returns:
            np.ndarray: Maximum excess demand for each firm
        """
        return self.functions["excess_demand"].set_maximum_excess_demand(
            current_production=self.ts.current("production"),
            target_production=self.ts.current("target_production"),
            limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
            limiting_labour_inputs=self.ts.current("labour_inputs"),
        )

    def prepare_buying_goods(
        self,
        previous_good_prices: np.ndarray,
        expected_inflation: float,
        assume_zero_growth: bool = False,
    ) -> None:
        """Prepare firms' buying plans for goods market.

        Sets targets for:
        - Intermediate input purchases
        - Capital input purchases
        Considers:
        - Previous prices
        - Expected inflation
        - Growth assumptions

        Args:
            previous_good_prices (np.ndarray): Previous period prices
            expected_inflation (float): Expected inflation rate
            assume_zero_growth (bool, optional): Whether to assume no growth. Defaults to False.
        """
        # Target intermediate inputs
        if assume_zero_growth:
            self.ts.target_intermediate_inputs.append(self.ts.initial("target_intermediate_inputs"))
        else:
            self.ts.target_intermediate_inputs.append(
                self.functions["target_intermediate_inputs"].compute_target_intermediate_inputs(
                    unconstrained_target_intermediate_inputs=self.ts.current(
                        "unconstrained_target_intermediate_inputs"
                    ),
                    target_short_term_credit=self.ts.current("target_short_term_credit"),
                    received_short_term_credit=self.ts.current("received_short_term_credit"),
                    previous_good_prices=previous_good_prices,
                    expected_inflation=expected_inflation,
                )
            )

        # Target capital inputs
        if assume_zero_growth:
            self.ts.target_capital_inputs.append(self.ts.initial("target_capital_inputs"))
        else:
            self.ts.target_capital_inputs.append(
                self.functions["target_capital_inputs"].compute_target_capital_inputs(
                    unconstrained_target_capital_inputs=self.ts.current("unconstrained_target_capital_inputs"),
                    target_long_term_credit=self.ts.current("target_long_term_credit"),
                    received_long_term_credit=self.ts.current("received_long_term_credit"),
                    previous_good_prices=previous_good_prices,
                    expected_inflation=expected_inflation,
                )
            )

        # Setting total real amount of goods to buy
        self.set_goods_to_buy(self.ts.current("target_intermediate_inputs") + self.ts.current("target_capital_inputs"))

    def prepare_selling_goods(self) -> None:
        """Prepare firms' selling plans for goods market.

        Sets up:
        - Quantities to sell (production + inventory)
        - Prices in USD
        - Industry classifications
        - Maximum excess demand
        """
        self.set_goods_to_sell(self.ts.current("production") + self.ts.current("inventory"))
        self.ts.price_in_usd.append(1.0 / self.exchange_rate_usd_to_lcu * self.ts.current("price"))
        self.ts.price_offered.append(self.compute_offered_price())
        self.set_prices(self.ts.current("price_in_usd"))
        self.set_seller_industries(self.states["Industry"])
        self.set_maximum_excess_demand(self.compute_maximum_excess_demand())

    def prepare_goods_market_clearing(
        self,
        exchange_rate_usd_to_lcu: float,
        previous_good_prices: np.ndarray,
        expected_inflation: float,
    ) -> None:
        """Prepare all aspects of goods market participation.

        Coordinates:
        - Exchange rate setting
        - Buying plans
        - Selling plans

        Args:
            exchange_rate_usd_to_lcu (float): Exchange rate USD to local currency
            previous_good_prices (np.ndarray): Previous period prices
            expected_inflation (float): Expected inflation rate
        """
        self.set_exchange_rate(exchange_rate_usd_to_lcu)
        self.prepare_buying_goods(
            previous_good_prices=previous_good_prices,
            expected_inflation=expected_inflation,
        )
        self.prepare_selling_goods()

    def distribute_bought_goods(self) -> None:
        """Distribute purchased goods between intermediate and capital inputs.

        Allocates actual purchases based on:
        - Desired intermediate inputs
        - Desired investment
        - Actual quantities bought
        """
        (
            new_intermediate_inputs,
            new_capital_inputs,
        ) = self.functions["bought_goods_distributor"].distribute_bought_goods(
            desired_intermediate_inputs=self.ts.current("target_intermediate_inputs"),
            desired_investment=self.ts.current("target_capital_inputs"),
            buy_real=self.ts.current("real_amount_bought"),
        )
        self.ts.real_amount_bought_as_intermediate_inputs.append(new_intermediate_inputs)
        self.ts.real_amount_bought_as_capital_goods.append(new_capital_inputs)
        """
        print(
            "DBG",
            list(
                (
                    (self.ts.current("target_intermediate_inputs") - new_intermediate_inputs).sum(axis=0)
                    + (self.ts.current("target_capital_inputs") - new_capital_inputs).sum(axis=0)
                ).round(2)
            ),
        )
        """

    def compute_gross_fixed_capital_formation(self, current_good_prices: np.ndarray) -> np.ndarray:
        """Calculate gross fixed capital formation.

        Computes value of capital goods purchases at current prices.

        Args:
            current_good_prices (np.ndarray): Current prices for valuation

        Returns:
            np.ndarray: Value of capital formation by industry
        """
        return (self.ts.current("real_amount_bought_as_capital_goods") * current_good_prices).sum(axis=0)

    def update_total_newly_bought_costs(self, current_good_prices: np.ndarray) -> None:
        """Update costs of newly purchased inputs.

        Allocates purchase costs between:
        - Intermediate inputs
        - Capital inputs
        Based on relative quantities at current prices.

        Args:
            current_good_prices (np.ndarray): Current prices for valuation
        """
        amount_ii = (self.ts.current("real_amount_bought_as_intermediate_inputs") * current_good_prices).sum(axis=1)
        amount_cap = (self.ts.current("real_amount_bought_as_capital_goods") * current_good_prices).sum(axis=1)

        # Just take fractions
        self.ts.total_intermediate_inputs_bought_costs.append(
            self.ts.current("nominal_amount_spent_in_lcu").sum(axis=1)
            * np.divide(
                amount_ii,
                amount_ii + amount_cap,
                out=np.zeros(amount_ii.shape),
                where=amount_ii + amount_cap != 0,
            )
        )
        self.ts.total_capital_inputs_bought_costs.append(
            self.ts.current("nominal_amount_spent_in_lcu").sum(axis=1)
            - self.ts.current("total_intermediate_inputs_bought_costs")
        )

    def compute_demand(self) -> np.ndarray:
        """Calculate realized demand for each firm's output.

        Combines:
        - Actual sales
        - Excess demand

        Returns:
            np.ndarray: Total demand for each firm
        """
        return self.functions["demand_for_goods"].compute_demand(
            sell_real=self.ts.current("real_amount_sold"),
            excess_demand=self.ts.current("real_excess_demand"),
        )

    def compute_nominal_production(self, current_good_prices: np.ndarray) -> np.ndarray:
        """Calculate nominal value of production.

        Args:
            current_good_prices (np.ndarray): Current prices for valuation

        Returns:
            np.ndarray: Nominal production value for each firm
        """
        return current_good_prices[self.states["Industry"]] * self.ts.current("production")

    def compute_inventory(self) -> np.ndarray:
        """Calculate end-of-period inventory levels.

        Considers:
        - Previous inventory
        - Current production
        - Sales
        - Depreciation

        Returns:
            np.ndarray: Updated inventory levels for each firm
        """
        return (1 - np.array(self.depreciation_rates)[self.states["Industry"]]) * np.maximum(
            0.0,
            self.ts.current("inventory") + self.ts.current("production") - self.ts.current("real_amount_sold"),
        )

    def compute_nominal_inventory(self, current_good_prices: np.ndarray) -> np.ndarray:
        """Calculate nominal value of inventories.

        Args:
            current_good_prices (np.ndarray): Current prices for valuation

        Returns:
            np.ndarray: Nominal inventory value for each firm
        """
        return current_good_prices[self.states["Industry"]] * self.ts.current("inventory")

    def compute_used_intermediate_inputs(self):
        """Calculate intermediate inputs used in production.

        Determines actual intermediate input usage based on:
        - Realized production
        - Input-output coefficients
        - Available input stocks
        - Input criticality requirements

        Returns:
            np.ndarray: Used intermediate inputs for each firm
        """
        return self.functions["production"].compute_intermediate_inputs_used(
            realised_production=self.ts.current("production"),
            intermediate_inputs_productivity_matrix=self.intermediate_inputs_productivity_matrix[
                :, self.states["Industry"]
            ].T,
            intermediate_inputs_stock=self.ts.current("intermediate_inputs_stock"),
            goods_criticality_matrix=self.goods_criticality_matrix,
        )

    def compute_used_intermediate_inputs_costs(self, current_good_prices: np.ndarray) -> np.ndarray:
        """Calculate cost of intermediate inputs used in production.

        Args:
            current_good_prices (np.ndarray): Current prices for inputs

        Returns:
            np.ndarray: Cost of used intermediate inputs for each firm
        """
        return (self.ts.current("used_intermediate_inputs") * current_good_prices).sum(axis=1)

    def compute_intermediate_inputs_stock(self) -> np.ndarray:
        """Calculate end-of-period intermediate input stocks.

        Updates stocks based on:
        - Previous stocks
        - Inputs used in production
        - New purchases

        Returns:
            np.ndarray: Updated intermediate input stocks
        """
        return np.maximum(
            0.0,
            self.ts.current("intermediate_inputs_stock")
            - self.ts.current("used_intermediate_inputs")
            + self.ts.current("real_amount_bought_as_intermediate_inputs"),
        )

    def compute_intermediate_inputs_stock_value(self, current_good_prices: np.ndarray) -> np.ndarray:
        """Calculate value of intermediate input stocks.

        Args:
            current_good_prices (np.ndarray): Current prices for valuation

        Returns:
            np.ndarray: Value of intermediate input stocks for each firm
        """
        return (self.ts.current("intermediate_inputs_stock") * current_good_prices).sum(axis=1)

    def compute_used_capital_inputs(self):
        """Calculate capital inputs used in production.

        Determines actual capital input usage based on:
        - Realized production
        - Capital productivity coefficients
        - Available capital stocks
        - Input criticality requirements

        Returns:
            np.ndarray: Used capital inputs for each firm
        """
        return self.functions["production"].compute_capital_inputs_used(
            realised_production=self.ts.current("production"),
            capital_inputs_depreciation_matrix=self.capital_inputs_depreciation_matrix[:, self.states["Industry"]].T,
            capital_inputs_stock=self.ts.current("capital_inputs_stock"),
            goods_criticality_matrix=self.goods_criticality_matrix,
        )

    def compute_used_capital_inputs_costs(self, current_good_prices: np.ndarray) -> np.ndarray:
        """Calculate cost of capital inputs used in production.

        Args:
            current_good_prices (np.ndarray): Current prices for inputs

        Returns:
            np.ndarray: Cost of used capital inputs for each firm
        """
        return (self.ts.current("used_capital_inputs") * current_good_prices).sum(axis=1)

    def compute_expected_capital_inputs_stock_value(
        self,
        current_good_prices: np.ndarray,
        estimated_inflation: float,
    ) -> np.ndarray:
        """Calculate expected future value of capital input stocks.

        Estimates future value considering:
        - Current stock levels
        - Current prices
        - Expected inflation

        Args:
            current_good_prices (np.ndarray): Current prices for valuation
            estimated_inflation (float): Expected inflation rate

        Returns:
            np.ndarray: Expected value of capital input stocks for each firm
        """
        return (1 + estimated_inflation) * (self.ts.current("capital_inputs_stock") * current_good_prices).sum(axis=1)

    def compute_capital_inputs_stock(self) -> np.ndarray:
        """Calculate end-of-period capital input stocks.

        Updates stocks considering:
        - Previous stocks
        - Used inputs
        - New purchases
        - Implementation delays

        Returns:
            np.ndarray: Updated capital input stocks
        """
        hist_bought_capital = np.array(self.ts.historic("real_amount_bought_as_capital_goods")[1:])
        delayed_bought_capital = np.zeros((self.ts.current("n_firms"), self.n_industries))
        for g in range(self.n_industries):
            delay = self.capital_inputs_delay[g]
            if delay < hist_bought_capital.shape[0]:
                delayed_bought_capital[:, g] = hist_bought_capital[-delay - 1, :, g]

        return np.maximum(
            0.0,
            self.ts.current("capital_inputs_stock") - self.ts.current("used_capital_inputs") + delayed_bought_capital,
        )

    def compute_capital_inputs_stock_value(self, current_good_prices: np.ndarray) -> np.ndarray:
        """Calculate value of capital input stocks.

        Args:
            current_good_prices (np.ndarray): Current prices for valuation

        Returns:
            np.ndarray: Value of capital input stocks for each firm
        """
        return (self.ts.current("capital_inputs_stock") * current_good_prices).sum(axis=1)

    def compute_total_inventory_change(self) -> np.ndarray:
        """Calculate nominal change in inventory value.

        Returns:
            np.ndarray: Change in inventory value for each firm
        """
        return self.ts.current("price") * (self.ts.current("inventory") - self.ts.prev("inventory"))

    def compute_taxes_paid_on_production(self, taxes_less_subsidies_rates: np.ndarray) -> np.ndarray:
        """Calculate taxes paid on production.

        Args:
            taxes_less_subsidies_rates (np.ndarray): Net tax rates by industry

        Returns:
            np.ndarray: Production taxes paid by each firm
        """
        return (
            taxes_less_subsidies_rates[self.states["Industry"]]
            * self.ts.current("production")
            * self.ts.current("price")
        )

    def compute_profits(self) -> np.ndarray:
        """Calculate profits for each firm.

        Computes:
        Revenue
        - Wages
        - Input costs
        - Production taxes
        - Interest

        Returns:
            np.ndarray: Profits for each firm
        """
        return (
            self.ts.current("price") * self.ts.current("production")
            - self.ts.current("total_wage")
            - self.ts.current("used_intermediate_inputs_costs")
            - self.ts.current("used_capital_inputs_costs")
            - self.ts.current("taxes_paid_on_production")
            - self.ts.current("interest_paid")
        )

    def compute_unit_costs(self) -> np.ndarray:
        """Calculate unit costs of production.

        Includes:
        - Wages
        - Input costs
        - Production taxes
        Per unit of output

        Returns:
            np.ndarray: Unit costs for each firm
        """
        return np.divide(
            self.ts.current("total_wage")
            + self.ts.current("used_intermediate_inputs_costs")
            + self.ts.current("used_capital_inputs_costs")
            + self.ts.current("taxes_paid_on_production"),
            self.ts.current("production"),
            out=np.zeros_like(self.ts.current("production")),
            where=self.ts.current("production") != 0.0,
        )

    def compute_corporate_taxes_paid(self, tau_firm: float) -> np.ndarray:
        """Calculate corporate income taxes.

        Args:
            tau_firm (float): Corporate tax rate

        Returns:
            np.ndarray: Corporate taxes paid by each firm
        """
        return tau_firm * np.maximum(0.0, self.ts.current("profits"))

    def compute_deposits(self) -> np.ndarray:
        """Calculate end-of-period deposit balances.

        Updates deposits based on:
        - Previous balance
        - Sales revenue
        - Costs and expenses
        - Taxes
        - Credit flows

        Returns:
            np.ndarray: Updated deposit balances for each firm
        """
        return (
            self.ts.current("deposits")
            + self.ts.current("nominal_amount_sold_in_lcu")
            - self.ts.current("total_wage")
            - self.ts.current("used_intermediate_inputs_costs")
            - self.ts.current("used_capital_inputs_costs")
            - self.ts.current("taxes_paid_on_production")
            - self.ts.current("corporate_taxes_paid")
            - self.ts.current("interest_paid")
            + self.ts.current("received_credit")
            - self.ts.current("debt_installments")
        )

    def compute_gross_operating_surplus_mixed_income(self) -> np.ndarray:
        """Calculate gross operating surplus and mixed income.

        Computes operating surplus as:
        Revenue from sales
        + Change in inventories
        - Wages
        - Intermediate input costs
        - Production taxes

        Returns:
            np.ndarray: Gross operating surplus for each firm
        """
        return (
            self.ts.current("nominal_amount_sold_in_lcu")
            + self.ts.current("price") * (self.ts.current("inventory") - self.ts.prev("inventory"))
            - self.ts.current("total_wage")
            - self.ts.current("used_intermediate_inputs_costs")
            - self.ts.current("taxes_paid_on_production")
        )

    def handle_insolvency(self, credit_market: CreditMarket) -> float:
        """Process insolvent firms and compute non-performing loan ratios.

        Handles firms that become insolvent by:
        - Marking them as insolvent based on equity and deposits
        - Removing their outstanding loans
        - Zeroing their deposits and equity
        - Computing the non-performing loan ratio

        Args:
            credit_market (CreditMarket): Credit market for loan processing

        Returns:
            float: Non-performing loan ratio for firm loans
        """
        self.states["is_insolvent"] = self.functions["demography"].handle_firm_insolvency(
            current_firm_is_insolvent=self.states["is_insolvent"],
            current_firm_equity=self.ts.current("equity"),
            current_firm_deposits=self.ts.current("deposits"),
        )

        # Remove loans
        insolvent_firms = np.where(self.states["is_insolvent"])[0]
        bad_firm_loans = credit_market.remove_loans_to_firm(insolvent_firms)

        # Update deposits
        new_firm_deposits = self.ts.current("deposits")
        new_firm_deposits[self.states["is_insolvent"]] = 0.0
        self.ts.deposits.pop()
        self.ts.deposits.append(new_firm_deposits)

        # Update equity
        new_firm_equity = self.ts.current("equity")
        new_firm_equity[self.states["is_insolvent"]] = 0.0
        self.ts.equity.pop()
        self.ts.equity.append(new_firm_equity)

        # Calculate the NPL ratio for firm loans
        total_loans_granted = (
            credit_market.ts.current("total_outstanding_loans_granted_firms_short_term")[0]
            + credit_market.ts.current("total_outstanding_loans_granted_firms_long_term")[0]
        )
        if total_loans_granted == 0.0:
            return 0.0
        else:
            return bad_firm_loans / total_loans_granted

    def compute_equity(self, current_good_prices: np.ndarray) -> np.ndarray:
        """Calculate equity value for each firm.

        Computes firm equity as:
        Assets (inventory, materials, capital, deposits)
        - Liabilities (debt)

        Args:
            current_good_prices (np.ndarray): Current prices for valuation

        Returns:
            np.ndarray: Equity values for each firm
        """
        material = np.dot(self.ts.current("intermediate_inputs_stock"), current_good_prices)
        capital = np.dot(self.ts.current("capital_inputs_stock"), current_good_prices)
        return (
            self.ts.current("inventory") * self.ts.current("price")
            + material
            + capital
            + self.ts.current("deposits")
            - self.ts.current("debt")
        )

    def compute_insolvency_rate(self) -> tuple[float, np.ndarray]:
        """Calculate insolvency statistics.

        Returns:
            tuple[float, np.ndarray]: Overall insolvency rate and count by industry
        """
        firm_insolvency_rate = self.states["is_insolvent"].mean()
        num_insolvent_firms_by_sector = np.zeros(self.n_industries)
        for g in range(self.n_industries):
            num_insolvent_firms_by_sector[g] = np.sum(self.states["is_insolvent"][self.states["Industry"] == g])
        self.states["is_insolvent"] = np.full(self.ts.current("n_firms"), False)
        return firm_insolvency_rate, num_insolvent_firms_by_sector

    def compute_total_debt(self) -> float:
        """Calculate total debt across all firms.

        Returns:
            float: Aggregate firm debt
        """
        return self.ts.current("debt").sum()

    def update_emissions(self, readjusted_factors: np.ndarray, emitting_indices: list | np.ndarray):
        """Update emissions from production activities.

        Calculates emissions from:
        - Intermediate input use
        - Capital input use
        Tracks emissions by source (coal, gas, oil, refined products)

        Args:
            readjusted_factors (np.ndarray): Emission factors per unit
            emitting_indices (list | np.ndarray): Indices of emitting sectors
        """
        used_intermediate_inputs = self.compute_used_intermediate_inputs()
        used_capital_inputs = self.compute_used_capital_inputs()
        inputs_emissions = used_intermediate_inputs[:, emitting_indices] @ readjusted_factors
        capital_emissions = used_capital_inputs[:, emitting_indices] @ readjusted_factors
        refining_firms = self.states["Industry"] == emitting_indices[-1]
        inputs_emissions[refining_firms] = 0
        capital_emissions[refining_firms] = 0

        self.ts.inputs_emissions.append(inputs_emissions)
        self.ts.capital_emissions.append(capital_emissions)

        # disaggregate emissions
        inputs_emissions_disaggregated = used_intermediate_inputs[:, emitting_indices] * readjusted_factors
        capital_emissions_disaggregated = used_capital_inputs[:, emitting_indices] * readjusted_factors
        inputs_emissions_disaggregated[refining_firms] = 0
        capital_emissions_disaggregated[refining_firms] = 0

        self.ts.coal_inputs_emissions.append(inputs_emissions_disaggregated[:, 0])
        self.ts.gas_inputs_emissions.append(inputs_emissions_disaggregated[:, 1])
        self.ts.oil_inputs_emissions.append(inputs_emissions_disaggregated[:, 2])
        self.ts.refined_products_inputs_emissions.append(inputs_emissions_disaggregated[:, 3])
        self.ts.coal_capital_emissions.append(capital_emissions_disaggregated[:, 0])
        self.ts.gas_capital_emissions.append(capital_emissions_disaggregated[:, 1])
        self.ts.oil_capital_emissions.append(capital_emissions_disaggregated[:, 2])
        self.ts.refined_products_capital_emissions.append(capital_emissions_disaggregated[:, 3])

    def compute_total_deposits(self) -> float:
        """Calculate total deposits across all firms.

        Returns:
            float: Aggregate firm deposits
        """
        return self.ts.current("deposits").sum()

    def save_to_h5(self, group: h5py.Group):
        """Save firm data to HDF5 format.

        Saves:
        - Time series data
        - Industry classifications
        - Firm-industry mapping

        Args:
            group (h5py.Group): HDF5 group to save data into
        """
        self.ts.write_to_h5("firms", group)

        firms_group = group["firms"]

        # Save industries DataFrame under 'firms_group'
        industries_df = self.industries_dataframe

        # Create a subgroup for industries under 'firms_group'
        industries_group = firms_group.create_group("industries")

        # Save the DataFrame to the HDF5 group
        self._save_dataframe_to_h5(industries_df, industries_group)

    @staticmethod
    def _save_dataframe_to_h5(df: pd.DataFrame, group: h5py.Group):
        """
        Saves a DataFrame to an HDF5 group.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            group (h5py.Group): The HDF5 group to save data into.
        """
        # Save index
        group.create_dataset("Firm_ID", data=df.index.values, dtype="int")

        # Save industry names as variable-length UTF-8 strings
        dt = h5py.string_dtype(encoding="utf-8")
        industry_names = df["Industry"].values
        group.create_dataset("Industry", data=industry_names, dtype=dt)

    def total_sales(self):
        """Get aggregate sales across all firms.

        Returns:
            float: Total firm sales
        """
        return self.ts.get_aggregate("total_sales")

    def total_used_input_costs(self):
        """Get aggregate intermediate input costs.

        Returns:
            float: Total intermediate input costs
        """
        return self.ts.get_aggregate("used_intermediate_inputs_costs")

    def get_total_inputs_emissions(self):
        """Get total emissions from intermediate inputs.

        Returns:
            float: Total input-related emissions
        """
        return self.ts.get_aggregate("inputs_emissions")

    def get_disaggregated_input_emissions(self, input_name: str):
        """Get emissions for specific input type.

        Args:
            input_name (str): Name of input type

        Returns:
            float: Emissions for specified input
        """
        return self.ts.get_aggregate(f"{input_name}_inputs_emissions")

    def get_disaggregated_capital_emissions(self, input_name: str):
        """Get capital emissions for specific input type.

        Args:
            input_name (str): Name of input type

        Returns:
            float: Capital emissions for specified input
        """
        return self.ts.get_aggregate(f"{input_name}_capital_emissions")

    def get_total_capital_emissions(self):
        """Get total emissions from capital use.

        Returns:
            float: Total capital-related emissions
        """
        return self.ts.get_aggregate("capital_emissions")

    def total_bought_input_costs(self):
        """Get aggregate cost of purchased inputs.

        Returns:
            float: Total input purchase costs
        """
        return self.ts.get_aggregate("total_intermediate_inputs_bought_costs")

    def total_operating_surplus(self):
        """Get aggregate operating surplus.

        Returns:
            float: Total gross operating surplus
        """
        return self.ts.get_aggregate("gross_operating_surplus_mixed_income")

    def total_wages(self):
        """Get aggregate wage payments.

        Returns:
            float: Total wages paid
        """
        return self.ts.get_aggregate("total_wage")

    def total_inventory_change(self):
        """Get aggregate change in inventories.

        Returns:
            float: Total inventory value change
        """
        return self.ts.get_aggregate("total_inventory_change")

    def total_capital_bought(self):
        """Get aggregate capital purchases.

        Returns:
            float: Total capital goods bought
        """
        return self.ts.get_aggregate("total_capital_inputs_bought_costs")

    def total_production(self):
        """Get aggregate production.

        Returns:
            float: Total production quantity
        """
        return self.ts.get_aggregate("production")

    def total_profits(self):
        """Get aggregate profits.

        Returns:
            float: Total firm profits
        """
        return self.ts.get_aggregate("profits")

    def total_taxes_paid_on_production(self):
        """Get aggregate production taxes paid.

        Returns:
            float: Total production taxes
        """
        return self.ts.get_aggregate("taxes_paid_on_production")


def fillna(array: np.ndarray, value: float = 0):
    """Fill NaN values in an array with a specified value.

    Args:
        array (np.ndarray): Input array with potential NaN values.
        value (float, optional): Value to replace NaN. Defaults to 0.

    Returns:
        np.ndarray: Array with NaN values replaced.
    """
    return np.where(np.isnan(array), value, array)
