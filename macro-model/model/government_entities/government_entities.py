import numpy as np
import pandas as pd

from pathlib import Path

from model.agents.agent import Agent
from model.timeseries import TimeSeries
from model.goods_market.value_type import ValueType
from model.util.function_mapping import get_functions
from model.government_entities.government_entities_ts import (
    create_government_entities_timeseries,
)

from typing import Any, Optional


class GovernmentEntities(Agent):
    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        year: int,
        t_max: int,
        n_industries: int,
        n_transactors: int,
        functions: dict[str, Any],
        parameters: dict[str, Any],
        ts: TimeSeries,
        states: dict[str, float | np.ndarray | list[np.ndarray]],
    ):
        super().__init__(
            country_name,
            all_country_names,
            year,
            t_max,
            n_industries,
            n_industries,
            n_transactors,
            functions,
            parameters,
            ts,
            states,
            transactor_settings={
                "Buyer Value Type": ValueType.NOMINAL,
                "Seller Value Type": ValueType.NONE,
                "Buyer Priority": 0,
                "Seller Priority": 0,
            },
        )

    @classmethod
    def from_data(
        cls,
        country_name: str,
        all_country_names: list[str],
        year: int,
        t_max: int,
        n_industries: int,
        data: pd.DataFrame,
        number_of_entities: int,
        government_consumption_model: Optional[Any],
        config: dict[str, Any],
    ) -> "GovernmentEntities":
        # Get corresponding functions and parameters
        functions = get_functions(
            config["functions"],
            loc="model.government_entities",
            func_dir=Path(__file__).parent / "func",
        )
        if "parameters" in config.keys():
            parameters = config["parameters"].copy()
        else:
            parameters = {}

        # Create the corresponding time series object
        ts = create_government_entities_timeseries(
            data=data,
            n_government_entities=number_of_entities,
        )

        # Additional states
        states = {"government_consumption_model": government_consumption_model}

        return cls(
            country_name,
            all_country_names,
            year,
            t_max,
            n_industries,
            number_of_entities,
            functions,
            parameters,
            ts,
            states,
        )

    def prepare_buying_goods(self) -> None:
        self.ts.desired_consumption_in_lcu.append(
            (
                self.functions["consumption"].compute_target_consumption(
                    previous_desired_government_consumption=self.ts.current("desired_consumption_in_lcu"),
                    model=self.states["government_consumption_model"],
                )
            )
        )
        self.ts.desired_consumption_in_usd.append(
            1.0 / self.exchange_rate_usd_to_lcu * self.ts.current("desired_consumption_in_lcu")
        )
        single_entity_consumption = self.ts.current("desired_consumption_in_usd") / self.ts.current(
            "n_government_entities"
        )
        all_entity_consumption = np.tile(single_entity_consumption, (self.ts.current("n_government_entities"), 1))
        self.set_goods_to_buy(all_entity_consumption)

    def prepare_selling_goods(self, n_industries: int) -> None:
        self.set_goods_to_sell(np.zeros(n_industries))
        self.set_prices(np.zeros(n_industries))

    def prepare_goods_market_clearing(
        self,
        n_industries: int,
        exchange_rate_usd_to_lcu: float,
    ) -> None:
        self.set_exchange_rate(exchange_rate_usd_to_lcu)
        self.prepare_buying_goods()
        self.prepare_selling_goods(n_industries)

    def record_consumption(self) -> None:
        self.ts.consumption_in_usd.append(self.ts.current("nominal_amount_spent_in_usd").sum(axis=0))
        self.ts.consumption_in_lcu.append(self.exchange_rate_usd_to_lcu * self.ts.current("consumption_in_usd"))
        self.ts.total_consumption.append([self.ts.current("consumption_in_lcu").sum()])
