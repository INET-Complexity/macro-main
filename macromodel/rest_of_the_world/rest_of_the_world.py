"""Rest of the World implementation module.

This module implements the Rest of the World (ROW) component that represents
all external economies interacting with the modeled countries. It handles:

1. International Trade:
   - Export supply decisions
   - Import demand behavior
   - Price setting in international markets
   - Currency conversion

2. Economic Integration:
   - Trade flow adjustments
   - Price level convergence
   - Market clearing processes
   - Exchange rate effects

3. Dynamic Behavior:
   - Trade volume forecasting
   - Price adjustment mechanisms
   - Growth and inflation impacts
   - Expectation formation

The ROW component serves as the external sector in the model, providing
closure for international trade and ensuring consistent global accounting.
"""

from functools import reduce
from typing import Any

import h5py
import numpy as np
import pandas as pd

from macro_data import SyntheticRestOfTheWorld
from macromodel.agents.agent import Agent
from macromodel.configurations import RestOfTheWorldConfiguration
from macromodel.configurations.row_configuration import RestOfTheWorldParameters
from macromodel.markets.goods_market.value_type import ValueType
from macromodel.rest_of_the_world.rest_of_the_world_ts import (
    create_rest_of_the_world_timeseries,
)
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import functions_from_model, update_functions


class RestOfTheWorld(Agent):
    """Rest of the World economic agent.

    This class represents all external economies as a single agent that trades
    with the modeled countries. It manages international trade flows, price
    setting, and market interactions through:
    - Import and export decisions
    - International price determination
    - Trade flow adjustments
    - Market clearing participation

    Attributes:
        country_name (str): Associated country identifier
        all_country_names (list[str]): All countries in model
        n_industries (int): Number of industrial sectors
        functions (dict[str, Any]): Economic function implementations
        parameters (RestOfTheWorldParameters): Behavioral parameters
        forecasting_window (int): Periods for forecasting
        assume_zero_growth (bool): Whether to assume no growth
        assume_zero_noise (bool): Whether to suppress random variation
        configuration (RestOfTheWorldConfiguration): Model settings
    """

    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        n_importers: int,
        n_exporters_by_industry: np.ndarray,
        functions: dict[str, Any],
        ts: TimeSeries,
        parameters: RestOfTheWorldParameters,
        states: dict[str, float | np.ndarray | list[np.ndarray]],
        forecasting_window: int,
        assume_zero_growth: bool,
        assume_zero_noise: bool,
        configuration: RestOfTheWorldConfiguration,
    ):
        """Initialize Rest of the World agent.

        Args:
            country_name (str): Associated country identifier
            all_country_names (list[str]): All countries in model
            n_industries (int): Number of industrial sectors
            n_importers (int): Number of importing agents
            n_exporters_by_industry (np.ndarray): Exporters per industry
            functions (dict[str, Any]): Economic function implementations
            ts (TimeSeries): Time series data
            parameters (RestOfTheWorldParameters): Behavioral parameters
            states (dict): Initial state variables
            forecasting_window (int): Periods for forecasting
            assume_zero_growth (bool): Whether to assume no growth
            assume_zero_noise (bool): Whether to suppress random variation
            configuration (RestOfTheWorldConfiguration): Model settings
        """
        super().__init__(
            country_name=country_name,
            all_country_names=all_country_names,
            n_industries=n_industries,
            n_transactors_sell=int(n_exporters_by_industry.sum()),
            n_transactors_buy=n_importers,
            ts=ts,
            states=states,
            transactor_settings={
                "Buyer Value Type": ValueType.NOMINAL,
                "Seller Value Type": ValueType.REAL,
                "Buyer Priority": 0,
                "Seller Priority": 1,
            },
        )

        self.functions = functions
        self.parameters = parameters
        self.forecasting_window = forecasting_window
        self.assume_zero_growth = assume_zero_growth
        self.assume_zero_noise = assume_zero_noise
        self.configuration = configuration

    @classmethod
    def from_pickled_row(
        cls,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        synthetic_row: SyntheticRestOfTheWorld,
        configuration: RestOfTheWorldConfiguration,
        calibration_data_before: pd.DataFrame,
        calibration_data_during: pd.DataFrame,
    ) -> "RestOfTheWorld":
        """Create ROW instance from synthetic data.

        Factory method that constructs a RestOfTheWorld instance using
        synthetic data and calibration information.

        Args:
            country_name (str): Associated country identifier
            all_country_names (list[str]): All countries in model
            n_industries (int): Number of industrial sectors
            synthetic_row (SyntheticRestOfTheWorld): Synthetic ROW data
            configuration (RestOfTheWorldConfiguration): Model settings
            calibration_data_before (pd.DataFrame): Pre-period calibration data
            calibration_data_during (pd.DataFrame): During-period calibration data

        Returns:
            RestOfTheWorld: Initialized ROW instance
        """
        functions = functions_from_model(model=configuration.functions, loc="macromodel.rest_of_the_world")

        data = synthetic_row.row_data.astype(float)
        data.rename_axis("Industry", inplace=True)

        exogenous_real_imports_before = calibration_data_before[("ROW", "Real Imports (Value)")].values
        exogenous_real_exports_before = calibration_data_before[("ROW", "Real Exports (Value)")].values

        exogenous_real_imports_during = calibration_data_during[("ROW", "Real Imports (Value)")].values
        exogenous_real_exports_during = calibration_data_during[("ROW", "Real Exports (Value)")].values

        n_exporters_by_industry = synthetic_row.n_exporters_by_industry
        n_importers = synthetic_row.n_importers

        row_exports_model = synthetic_row.exports_model
        row_imports_model = synthetic_row.imports_model

        ts = create_rest_of_the_world_timeseries(
            data=data,
            n_industries=n_industries,
        )

        states = {
            "row_exports_model": row_exports_model,
            "row_imports_model": row_imports_model,
            "Industry": np.arange(n_industries),
            "number_of_exporters_by_industry": n_exporters_by_industry.astype(int),
            "exogenous_real_imports_before": exogenous_real_imports_before,
            "exogenous_real_imports_during": exogenous_real_imports_during,
            "exogenous_real_exports_before": exogenous_real_exports_before,
            "exogenous_real_exports_during": exogenous_real_exports_during,
        }

        return cls(
            country_name=country_name,
            all_country_names=all_country_names,
            n_industries=n_industries,
            functions=functions,
            ts=ts,
            states=states,
            n_importers=n_importers,
            n_exporters_by_industry=n_exporters_by_industry,
            parameters=configuration.parameters,
            forecasting_window=configuration.forecasting_window,
            assume_zero_growth=configuration.assume_zero_growth,
            assume_zero_noise=configuration.assume_zero_noise,
            configuration=configuration,
        )

    def reset(self, configuration: RestOfTheWorldConfiguration) -> None:
        """Reset ROW state with new configuration.

        Args:
            configuration (RestOfTheWorldConfiguration): New model settings
        """
        self.gen_reset()
        update_functions(
            model=configuration.functions,
            loc="macromodel.rest_of_the_world",
            functions=self.functions,
            force_reset=["imports", "exports"],
        )
        self.parameters = configuration.parameters
        self.forecasting_window = configuration.forecasting_window
        self.assume_zero_growth = configuration.assume_zero_growth
        self.assume_zero_noise = configuration.assume_zero_noise
        self.configuration = configuration

    def estimate_inflation(self, average_country_ppi_inflation: float) -> float:
        """Estimate ROW inflation rate.

        Args:
            average_country_ppi_inflation (float): Average PPI inflation

        Returns:
            float: Estimated ROW inflation rate
        """
        return self.functions["inflation"].compute_inflation(
            average_country_ppi_inflation=average_country_ppi_inflation
        )

    def prepare_buying_goods(
        self,
        aggregate_country_production_index: float,
        aggregate_country_price_index: float,
    ) -> None:
        """Prepare import decisions.

        Determines desired import volumes based on historical data and
        current economic conditions.

        Args:
            aggregate_country_production_index (float): Production level
            aggregate_country_price_index (float): Price level
        """
        historic_total_real_imports = np.concatenate(
            (
                self.states["exogenous_real_imports_before"][-self.forecasting_window :],
                np.sum(
                    np.array(self.ts.historic("imports_in_lcu")) / np.array(self.ts.historic("price_in_lcu")),
                    axis=1,
                ),
            )
        )
        if self.assume_zero_growth:
            self.ts.desired_imports_in_lcu.append(self.ts.initial("desired_imports_in_lcu"))
        else:
            self.ts.desired_imports_in_lcu.append(
                self.functions["imports"].compute_imports(
                    historic_total_real_imports=historic_total_real_imports,
                    historic_total_real_imports_during=self.states["exogenous_real_imports_during"],
                    current_time=len(self.ts.historic("total_exports")),
                    initial_desired_imports=self.ts.initial("desired_imports_in_lcu"),
                    model=self.states["row_imports_model"],
                    aggregate_country_production_index=aggregate_country_production_index,
                    aggregate_country_price_index=aggregate_country_price_index,
                    adjustment_speed=self.parameters.adjustment_speed,
                    assume_zero_noise=self.assume_zero_noise,
                )
            )
        self.ts.desired_imports_in_usd.append(
            1.0 / self.exchange_rate_usd_to_lcu * self.ts.current("desired_imports_in_lcu")
        )
        assert np.all(self.ts.current("desired_imports_in_usd") >= 0.0)
        self.set_goods_to_buy(
            np.stack(
                [
                    self.ts.current("desired_imports_in_usd") / self.n_transactors_buy
                    for _ in range(self.n_transactors_buy)
                ]
            )
        )

    def prepare_selling_goods(
        self,
        aggregate_country_production_index: float,
        aggregate_country_price_index: float,
    ) -> None:
        """Prepare export decisions.

        Determines desired export volumes and prices based on historical
        data and current economic conditions.

        Args:
            aggregate_country_production_index (float): Production level
            aggregate_country_price_index (float): Price level
        """
        # Set desired exports
        historic_total_real_exports = np.concatenate(
            (
                self.states["exogenous_real_exports_before"][-self.forecasting_window :],
                np.array(self.ts.historic("total_exports")).flatten(),
            )
        )
        if self.assume_zero_growth:
            self.ts.desired_exports_real.append(self.ts.initial("desired_exports_real"))
        else:
            self.ts.desired_exports_real.append(
                self.functions["exports"].compute_exports(
                    historic_total_real_exports=historic_total_real_exports,
                    historic_total_real_exports_during=self.states["exogenous_real_exports_during"],
                    current_time=len(self.ts.historic("total_exports")),
                    initial_desired_exports=self.ts.initial("desired_exports_real"),
                    model=self.states["row_exports_model"],
                    aggregate_country_production_index=aggregate_country_production_index,
                    adjustment_speed=self.parameters.adjustment_speed,
                    assume_zero_noise=self.assume_zero_noise,
                )
            )
        assert np.all(self.ts.current("desired_exports_real") >= 0.0)
        self.set_goods_to_sell(
            np.array(
                reduce(
                    lambda a, b: a + b,
                    (
                        [
                            self.ts.current("desired_exports_real")[industry]
                            / self.states["number_of_exporters_by_industry"][industry]
                        ]
                        * s
                        for industry, s in zip(
                            range(self.n_industries),
                            list(self.states["number_of_exporters_by_industry"]),
                        )
                    ),
                )
            )
        )

        # Set prices
        self.ts.price_in_lcu.append(
            self.functions["prices"].compute_price(
                initial_price=self.ts.initial("price_in_lcu"),
                aggregate_country_price_index=aggregate_country_price_index,
                adjustment_speed=self.parameters.adjustment_speed,
            )
        )
        self.ts.price_in_usd.append(1.0 / self.exchange_rate_usd_to_lcu * self.ts.current("price_in_lcu"))
        assert np.all(self.ts.current("price_in_usd") > 0.0)
        self.ts.price_offered.append(self.ts.current("price_in_usd"))
        self.set_prices(self.ts.current("price_in_usd")[self.states["Industry"]])

        # Seller industries
        self.set_seller_industries(self.states["Industry"])

        # Excess demand
        self.set_maximum_excess_demand(
            self.functions["excess_demand"].set_maximum_excess_demand(
                n_exporters=self.states["number_of_exporters_by_industry"].sum(),
            )
        )

    def prepare_goods_market_clearing(
        self,
        aggregate_country_production_index: float,
        aggregate_country_price_index: float,
    ) -> None:
        """Prepare for goods market clearing.

        Sets up all necessary trade decisions and prices for market clearing.

        Args:
            aggregate_country_production_index (float): Production level
            aggregate_country_price_index (float): Price level
        """
        self.set_exchange_rate(1.0)
        self.prepare_buying_goods(
            aggregate_country_production_index=aggregate_country_production_index,
            aggregate_country_price_index=aggregate_country_price_index,
        )
        self.prepare_selling_goods(
            aggregate_country_production_index=aggregate_country_production_index,
            aggregate_country_price_index=aggregate_country_price_index,
        )

    def update_planning_metrics(
        self,
        aggregate_country_production_index: float,
        aggregate_country_price_index: float,
    ) -> None:
        """Update planning metrics for market participation.

        Args:
            aggregate_country_production_index (float): Production level
            aggregate_country_price_index (float): Price level
        """
        self.prepare_goods_market_clearing(
            aggregate_country_production_index=aggregate_country_production_index,
            aggregate_country_price_index=aggregate_country_price_index,
        )

    def record_bought_goods(self) -> None:
        """Record results of goods market transactions.

        Updates time series with actual trade volumes and values.
        """
        self.ts.exports_real.append(self.ts.current("real_amount_sold"))
        self.ts.total_exports.append([self.ts.current("exports_real").sum()])
        self.ts.imports_in_usd.append(self.ts.current("nominal_amount_spent_in_lcu")[0])
        self.ts.imports_in_lcu.append(self.exchange_rate_usd_to_lcu * self.ts.current("imports_in_usd"))
        self.ts.total_imports.append([self.ts.current("imports_in_lcu").sum()])

    def save_to_h5(self, file: h5py.File) -> None:
        """Save ROW data to HDF5 file.

        Args:
            file (h5py.File): HDF5 file to save to
        """
        group = file.create_group("ROW")
        self.ts.write_to_h5("rest_of_the_world", group)
