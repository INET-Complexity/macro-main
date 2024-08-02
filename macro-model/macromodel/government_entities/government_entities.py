import h5py
import numpy as np
from macro_data import SyntheticGovernmentEntities
from typing import Any, Optional

from macromodel.configurations import GovernmentEntitiesConfiguration
from macromodel.agents.agent import Agent
from macromodel.goods_market.value_type import ValueType
from macromodel.government_entities.government_entities_ts import (
    create_government_entities_timeseries,
)
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import functions_from_model, update_functions


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
    ):
        functions = functions_from_model(model=configuration.functions, loc="macromodel.government_entities")

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

    def reset(self, configuration: GovernmentEntitiesConfiguration):
        self.gen_reset()
        update_functions(
            model=configuration.functions,
            loc="macromodel.government_entities",
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

    def record_consumption(self) -> None:
        self.ts.consumption_in_usd.append(self.ts.current("nominal_amount_spent_in_usd").sum(axis=0))
        self.ts.consumption_in_lcu.append(self.exchange_rate_usd_to_lcu * self.ts.current("consumption_in_usd"))
        self.ts.total_consumption.append([self.ts.current("consumption_in_lcu").sum()])

    def save_to_h5(self, group: h5py.Group):
        self.ts.write_to_h5("government_entities", group)

    def total_consumption(self):
        return self.ts.get_aggregate("total_consumption")
