import h5py
import numpy as np
import pandas as pd
from macro_data import SyntheticRestOfTheWorld
from pathlib import Path
from typing import Any, Optional

from macromodel.configurations import RestOfTheWorldConfiguration
from macromodel.agents.agent import Agent
from macromodel.goods_market.value_type import ValueType
from macromodel.rest_of_the_world.rest_of_the_world_ts import create_rest_of_the_world_timeseries
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import get_functions, functions_from_model


class RestOfTheWorld(Agent):
    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        functions: dict[str, Any],
        ts: TimeSeries,
        states: dict[str, float | np.ndarray | list[np.ndarray]],
    ):
        super().__init__(
            country_name,
            all_country_names,
            n_industries,
            n_industries,
            1,
            ts,
            states,
            transactor_settings={
                "Buyer Value Type": ValueType.NOMINAL,
                "Seller Value Type": ValueType.REAL,
                "Buyer Priority": 1,
                "Seller Priority": 1,
            },
        )

        self.functions = functions

    @classmethod
    def from_pickled_row(
        cls,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        synthetic_row: SyntheticRestOfTheWorld,
        configuration: RestOfTheWorldConfiguration,
        average_ppi_inflation: float,
    ) -> "RestOfTheWorld":
        functions = functions_from_model(model=configuration.functions, loc="inet_macromodel.rest_of_the_world")

        data = synthetic_row.row_data.astype(float)
        data.rename_axis("Industry", inplace=True)

        row_exports_model = synthetic_row.exports_model
        row_imports_model = synthetic_row.imports_model

        ts = create_rest_of_the_world_timeseries(
            data=data,
            initial_inflation=functions["inflation"].compute_inflation(
                average_country_ppi_inflation=average_ppi_inflation
            ),
            n_industries=n_industries,
        )

        states = {
            "row_exports_model": row_exports_model,
            "row_imports_model": row_imports_model,
            "Industry": np.arange(n_industries),
        }

        return cls(
            country_name,
            all_country_names,
            n_industries,
            functions,
            ts,
            states,
        )

    @classmethod
    def from_data(
        cls,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        data: pd.DataFrame,
        row_exports_model: Optional[Any],
        row_imports_model: Optional[Any],
        average_country_ppi_inflation: float,
        config: dict[str, Any],
    ) -> "RestOfTheWorld":
        # Get corresponding functions and parameters
        functions = get_functions(
            config["functions"],
            loc="inet_macromodel.rest_of_the_world",
            func_dir=Path(__file__).parent / "func",
        )
        if "parameters" in config.keys():
            parameters = config["parameters"].copy()
        else:
            parameters = {}

        # Create the corresponding time series object
        ts = create_rest_of_the_world_timeseries(
            data=data,
            initial_inflation=functions["inflation"].compute_inflation(
                average_country_ppi_inflation=average_country_ppi_inflation
            ),
            n_industries=n_industries,
        )

        # Additional states
        states = {
            "row_exports_model": row_exports_model,
            "row_imports_model": row_imports_model,
            "Industry": list(range(n_industries)),
        }

        return cls(
            country_name,
            all_country_names,
            n_industries,
            functions,
            ts,
            states,
        )

    def estimate_inflation(self, average_country_ppi_inflation: float) -> float:
        return self.functions["inflation"].compute_inflation(
            average_country_ppi_inflation=average_country_ppi_inflation
        )

    def prepare_buying_goods(self) -> None:
        self.ts.desired_imports_in_lcu.append(
            self.functions["imports"].compute_imports(
                previous_desired_imports=self.ts.current("desired_imports_in_lcu"),
                model=self.states["row_imports_model"],
            )
        )
        self.ts.desired_imports_in_usd.append(
            1.0 / self.exchange_rate_usd_to_lcu * self.ts.current("desired_imports_in_lcu")
        )
        self.set_goods_to_buy(np.array([self.ts.current("desired_imports_in_usd")]))

    def prepare_selling_goods(self) -> None:
        # Set desired exports
        self.ts.desired_exports_real.append(
            self.functions["exports"].compute_exports(
                previous_desired_exports=self.ts.current("desired_exports_real"),
                model=self.states["row_exports_model"],
            )
        )
        self.set_goods_to_sell(self.ts.current("desired_exports_real"))

        # Set prices
        self.ts.price_in_lcu.append(
            self.functions["prices"].compute_price(
                previous_price=self.ts.current("price_in_lcu"),
                previous_row_inflation=self.ts.current("inflation")[0],
            )
        )

        self.ts.price_in_usd.append(1.0 / self.exchange_rate_usd_to_lcu * self.ts.current("price_in_lcu"))
        self.ts.price_offered.append(self.ts.current("price_in_usd"))
        self.set_prices(self.ts.current("price_in_usd"))

        # Seller industries
        self.set_seller_industries(np.arange(self.n_industries))

    def prepare_goods_market_clearing(self) -> None:
        self.set_exchange_rate(1.0)
        self.prepare_buying_goods()
        self.prepare_selling_goods()

    def update_planning_metrics(self, average_country_ppi_inflation: float) -> None:
        self.ts.inflation.append([self.estimate_inflation(average_country_ppi_inflation=average_country_ppi_inflation)])
        self.prepare_goods_market_clearing()

    def record_bought_goods(self) -> None:
        self.ts.exports_real.append(self.ts.current("real_amount_sold"))
        self.ts.imports_in_usd.append(self.ts.current("nominal_amount_spent_in_lcu")[0])
        self.ts.imports_in_lcu.append(self.exchange_rate_usd_to_lcu * self.ts.current("imports_in_usd"))

    def save_to_h5(self, file: h5py.File) -> None:
        group = file.create_group("ROW")
        self.ts.write_to_h5("rest_of_the_world", group)
