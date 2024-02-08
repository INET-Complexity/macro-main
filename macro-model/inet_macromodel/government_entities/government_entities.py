import h5py
import numpy as np
from inet_data import SyntheticGovernmentEntities
from typing import Any

from inet_macromodel.configurations import GovernmentEntitiesConfiguration
from inet_macromodel.agents.agent import Agent
from inet_macromodel.goods_market.value_type import ValueType
from inet_macromodel.government_entities.government_entities_ts import (
    create_government_entities_timeseries,
)
from inet_macromodel.timeseries import TimeSeries
from inet_macromodel.util.function_mapping import functions_from_model


class GovernmentEntities(Agent):
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
        super().__init__(
            country_name,
            all_country_names,
            n_industries,
            n_transactors,
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
    ):
        functions = functions_from_model(model=configuration.functions, loc="inet_macromodel.government_entities")

        ts = create_government_entities_timeseries(
            data=synthetic_government_entities.gov_entity_data,
            n_government_entities=synthetic_government_entities.number_of_entities,
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

    def save_to_h5(self, group: h5py.Group):
        # TODO : this is a temporary solution, we need to find a better way to save the data
        # the problem is that real amount sold somehow changes size, it is not clear why
        # in the test, we start with 39 government entities, but after the first iteration the
        # size of the real amount sold array is 18 (the number of industries)
        # keys_to_delete = [k for k in self.ts.dicts.keys() if "sold" in k]
        # for k in keys_to_delete:
        #     del self.ts.dicts[k]
        # self.ts.write_to_h5("government_entities", group)
        ...
