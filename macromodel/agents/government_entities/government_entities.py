"""Government entities implementation for macroeconomic modeling.

This module implements government entities that represent various government
organizations and agencies that participate in the economy through:
- Government consumption and investment
- Public sector spending
- Infrastructure development
- Public service provision

The entities operate in goods markets by:
- Planning consumption based on economic conditions
- Adjusting spending for inflation and growth
- Tracking emissions from consumption activities
- Managing multiple government organizations
"""

from typing import Any, Optional

import h5py
import numpy as np

from macro_data import SyntheticGovernmentEntities
from macromodel.agents.agent import Agent
from macromodel.agents.government_entities.government_entities_ts import (
    create_government_entities_timeseries,
)
from macromodel.configurations import GovernmentEntitiesConfiguration
from macromodel.markets.goods_market.value_type import ValueType
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import functions_from_model, update_functions


class GovernmentEntities(Agent):
    """Government entities that consume and invest in the economy.

    This class represents multiple government organizations that participate
    in economic activity through consumption and investment. It manages:
    - Consumption planning and execution
    - Price-adjusted spending
    - Growth and inflation expectations
    - Emissions tracking from consumption
    - Multiple entity coordination

    The entities operate as buyers in goods markets with:
    - Consumption targets based on economic conditions
    - Price level adjustments
    - Growth and inflation expectations
    - Historical consumption patterns
    - Optional emissions tracking

    Attributes:
        functions (dict[str, Any]): Function implementations for operations
        states (dict[str, Any]): State variables including consumption models
        ts (TimeSeries): Time series data tracking consumption and emissions
    """

    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        n_transactors: int,
        functions: dict[str, Any],
        ts: TimeSeries,
        states: dict[str, Any],
    ):
        """Initialize government entities.

        Args:
            country_name (str): Name of the country these entities serve
            all_country_names (list[str]): List of all countries in model
            n_industries (int): Number of industries in the economy
            n_transactors (int): Number of government entities
            functions (dict[str, Any]): Function implementations
            ts (TimeSeries): Time series for tracking variables
            states (dict[str, Any]): State variables and parameters
        """
        super().__init__(
            country_name,
            all_country_names,
            n_industries,
            n_industries,
            n_transactors,
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

    @classmethod
    def from_pickled_agent(
        cls,
        synthetic_government_entities: SyntheticGovernmentEntities,
        configuration: GovernmentEntitiesConfiguration,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        add_emissions: bool = False,
        emission_factors_lcu: Optional[np.ndarray] = None,
        emitting_indices: Optional[np.ndarray] = None,
    ):
        """Create government entities from pickled data.

        Initializes entities with:
        - Consumption functions from configuration
        - Time series from synthetic data
        - Optional emissions tracking setup
        - Country-specific parameters

        Args:
            synthetic_government_entities (SyntheticGovernmentEntities):
                Synthetic data with initial states and parameters
            configuration (GovernmentEntitiesConfiguration):
                Configuration for entity operations
            country_name (str): Name of the country these entities serve
            all_country_names (list[str]): List of all countries
            n_industries (int): Number of industries
            add_emissions (bool, optional): Whether to track emissions
            emission_factors_lcu (np.ndarray, optional): Emission factors
                per unit of consumption in local currency
            emitting_indices (np.ndarray, optional): Indices of goods
                that generate emissions

        Returns:
            GovernmentEntities: Initialized government entities
        """
        functions = functions_from_model(model=configuration.functions, loc="macromodel.agents.government_entities")

        ts = create_government_entities_timeseries(
            data=synthetic_government_entities.gov_entity_data,
            n_government_entities=synthetic_government_entities.number_of_entities,
            add_emissions=add_emissions,
            emission_factors_lcu=emission_factors_lcu,
            emitting_indices=emitting_indices,
        )

        states = {"government_consumption_model": synthetic_government_entities.government_consumption_model}

        return cls(
            country_name=country_name,
            all_country_names=all_country_names,
            n_industries=n_industries,
            n_transactors=synthetic_government_entities.number_of_entities,
            functions=functions,
            ts=ts,
            states=states,
        )

    def reset(self, configuration: GovernmentEntitiesConfiguration):
        """Reset government entities to initial state.

        Resets all state variables and updates function implementations
        based on the provided configuration.

        Args:
            configuration (GovernmentEntitiesConfiguration): New
                configuration parameters for the reset state
        """
        self.gen_reset()
        update_functions(
            model=configuration.functions,
            loc="macromodel.agents.government_entities",
            functions=self.functions,
            force_reset=["consumption"],
        )

    def prepare_buying_goods(
        self,
        exogenous_gov_consumption_before: Optional[np.ndarray],
        exogenous_gov_consumption_during: Optional[np.ndarray],
        initial_good_prices: np.ndarray,
        current_good_prices: np.ndarray,
        historic_ppi: np.ndarray,
        expected_growth: float,
        expected_inflation: float,
        forecasting_window: int,
        assume_zero_growth: bool,
        assume_zero_noise: bool,
    ) -> None:
        """Prepare government entities for goods market participation.

        Determines consumption targets considering:
        - Historical consumption patterns
        - Price level changes
        - Growth expectations
        - Inflation expectations
        - Optional exogenous consumption paths

        Args:
            exogenous_gov_consumption_before (np.ndarray, optional):
                Pre-specified consumption before current period
            exogenous_gov_consumption_during (np.ndarray, optional):
                Pre-specified consumption for current period
            initial_good_prices (np.ndarray): Initial price levels
            current_good_prices (np.ndarray): Current price levels
            historic_ppi (np.ndarray): Historical producer price index
            expected_growth (float): Expected economic growth rate
            expected_inflation (float): Expected inflation rate
            forecasting_window (int): Window for consumption forecasting
            assume_zero_growth (bool): Whether to assume no growth
            assume_zero_noise (bool): Whether to assume deterministic
                consumption paths
        """
        if exogenous_gov_consumption_before is None:
            historic_total_consumption = np.array(self.ts.historic("total_consumption")).flatten() / historic_ppi
        else:
            historic_total_consumption = np.concatenate(
                (
                    exogenous_gov_consumption_before[-forecasting_window:],
                    np.array(self.ts.historic("total_consumption")).flatten() / historic_ppi,
                )
            )
        if assume_zero_growth:
            self.ts.desired_consumption_in_lcu.append(self.ts.initial("consumption_in_lcu"))
        else:
            self.ts.desired_consumption_in_lcu.append(
                (
                    self.functions["consumption"].compute_target_consumption(
                        previous_desired_government_consumption=self.ts.current("desired_consumption_in_lcu"),
                        model=self.states["government_consumption_model"],
                        historic_total_consumption=historic_total_consumption,
                        initial_good_prices=initial_good_prices,
                        current_good_prices=current_good_prices,
                        expected_growth=expected_growth,
                        expected_inflation=expected_inflation,
                        current_time=len(self.ts.historic("consumption_in_usd")),
                        exogenous_total_consumption=exogenous_gov_consumption_during,
                        forecasting_window=forecasting_window,
                        assume_zero_noise=assume_zero_noise,
                    )
                )
            )
        self.ts.desired_consumption_in_usd.append(
            1.0 / self.exchange_rate_usd_to_lcu * self.ts.current("desired_consumption_in_lcu")
        )
        single_entity_consumption = self.ts.current("desired_consumption_in_usd") / self.ts.current(
            "n_government_entities"
        )
        all_entity_consumption = np.tile(
            single_entity_consumption,
            (self.ts.current("n_government_entities"), 1),
        )
        self.set_goods_to_buy(all_entity_consumption)

    def prepare_selling_goods(self, n_industries: int) -> None:
        """Prepare selling side of goods market (not used).

        Government entities only participate as buyers.

        Args:
            n_industries (int): Number of industries
        """
        self.set_goods_to_sell(np.zeros(n_industries))
        self.set_prices(np.zeros(n_industries))

    def prepare_goods_market_clearing(
        self,
        n_industries: int,
        exchange_rate_usd_to_lcu: float,
        exogenous_gov_consumption_before: Optional[np.ndarray],
        exogenous_gov_consumption_during: Optional[np.ndarray],
        initial_good_prices: np.ndarray,
        current_good_prices: np.ndarray,
        historic_ppi: np.ndarray,
        expected_growth: float,
        expected_inflation: float,
        forecasting_window: int,
        assume_zero_growth: bool,
        assume_zero_noise: bool,
    ) -> None:
        """Prepare for goods market clearing.

        Sets up entities for market participation by:
        - Setting exchange rates
        - Preparing buying plans
        - Setting up selling side (empty)

        Args:
            [same as prepare_buying_goods]
            exchange_rate_usd_to_lcu (float): Exchange rate from USD to
                local currency
        """
        self.set_exchange_rate(exchange_rate_usd_to_lcu)
        self.prepare_buying_goods(
            exogenous_gov_consumption_before=exogenous_gov_consumption_before,
            exogenous_gov_consumption_during=exogenous_gov_consumption_during,
            initial_good_prices=initial_good_prices,
            current_good_prices=current_good_prices,
            historic_ppi=historic_ppi,
            expected_growth=expected_growth,
            expected_inflation=expected_inflation,
            forecasting_window=forecasting_window,
            assume_zero_growth=assume_zero_growth,
            assume_zero_noise=assume_zero_noise,
        )
        self.prepare_selling_goods(n_industries)

    def record_consumption(
        self,
        add_emissions: bool = False,
        readjusted_factors: Optional[np.ndarray] = None,
        emitting_indices: Optional[np.ndarray] = None,
    ) -> None:
        """Record consumption and optional emissions.

        Records:
        - Consumption in USD and local currency
        - Total consumption across entities
        - Optional emissions by type if tracking enabled

        Args:
            add_emissions (bool, optional): Whether to track emissions
            readjusted_factors (np.ndarray, optional): Emission factors
                adjusted for current conditions
            emitting_indices (np.ndarray, optional): Indices of goods
                that generate emissions
        """
        self.ts.consumption_in_usd.append(self.ts.current("nominal_amount_spent_in_usd").sum(axis=0))
        self.ts.consumption_in_lcu.append(self.exchange_rate_usd_to_lcu * self.ts.current("consumption_in_usd"))
        if add_emissions:
            emissions = np.sum(self.ts.current("consumption_in_lcu")[emitting_indices] * readjusted_factors).sum()
            self.ts.emissions.append(emissions)
            self.ts.coal_emissions.append(
                np.sum(self.ts.current("consumption_in_lcu")[emitting_indices] * readjusted_factors[0])
            )
            self.ts.gas_emissions.append(
                np.sum(self.ts.current("consumption_in_lcu")[emitting_indices] * readjusted_factors[1])
            )
            self.ts.oil_emissions.append(
                np.sum(self.ts.current("consumption_in_lcu")[emitting_indices] * readjusted_factors[2])
            )
            self.ts.refined_products_emissions.append(
                np.sum(self.ts.current("consumption_in_lcu")[emitting_indices] * readjusted_factors[3])
            )
        self.ts.total_consumption.append([self.ts.current("consumption_in_lcu").sum()])

    def save_to_h5(self, group: h5py.Group):
        """Save government entities data to HDF5.

        Stores all time series data in the specified HDF5 group.

        Args:
            group (h5py.Group): HDF5 group to save data in
        """
        self.ts.write_to_h5("government_entities", group)

    def total_consumption(self):
        """Get total consumption across all entities.

        Returns:
            np.ndarray: Historical total consumption values
        """
        return self.ts.get_aggregate("total_consumption")

    def emissions(self):
        """Get total emissions from consumption.

        Returns:
            np.ndarray: Historical emissions values
        """
        return self.ts.historic("emissions")

    def disaggregated_emissions(self, input_name: str):
        """Get emissions by input type.

        Args:
            input_name (str): Name of input type (e.g., 'coal', 'gas')

        Returns:
            np.ndarray: Historical emissions for specified input
        """
        return self.ts.historic(f"{input_name}_emissions")
