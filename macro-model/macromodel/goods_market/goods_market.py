import logging
import numpy as np
import pandas as pd
from typing import Any, Tuple, Optional

from macromodel.agents.agent import Agent
from macromodel.configurations import GoodsMarketConfiguration
from macromodel.goods_market.goods_market_ts import create_goods_market_timeseries
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import functions_from_model


SupplyChain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]]


class GoodsMarket:
    def __init__(
        self,
        n_industries: int,
        functions: dict[str, Any],
        trade_proportions: pd.DataFrame,
        ts: TimeSeries,
        goods_market_participants: dict[str, list[Agent]],
        states: dict[str, Any],
        buyer_priorities: dict[str, np.ndarray],
        seller_priorities: dict[str, np.ndarray],
    ):
        self.n_industries = n_industries
        self.functions = functions
        self.trade_proportions = trade_proportions.fillna(0.0)
        self.ts = ts
        self.goods_market_participants = goods_market_participants
        self.states = states
        self.buyer_priorities = buyer_priorities
        self.seller_priorities = seller_priorities

    @classmethod
    def from_data(
        cls,
        n_industries: int,
        trade_proportions: pd.DataFrame,
        configuration: GoodsMarketConfiguration,
        goods_market_participants: dict[str, list[Agent]],
        origin_trade_proportions: np.ndarray,
        destin_trade_proportions: np.ndarray,
        initial_supply_chain: Optional[SupplyChain] = None,
    ) -> "GoodsMarket":
        # Get corresponding functions
        functions = functions_from_model(configuration.functions, loc="macromodel.goods_market")
        n_countries = int(np.sqrt(len(origin_trade_proportions) / n_industries))

        states = {
            "origin_trade_proportions": origin_trade_proportions.reshape((n_countries, n_countries, n_industries)),
            "destin_trade_proportions": destin_trade_proportions.reshape((n_countries, n_countries, n_industries)),
            "previous_supply_chain": None,
        }

        # Create the corresponding time series object
        ts = create_goods_market_timeseries(n_industries)

        if initial_supply_chain is None:
            states["current_supply_chain"] = {g: {} for g in range(n_industries)}
        else:
            states["current_supply_chain"] = initial_supply_chain

        buyer_priorities = {}
        seller_priorities = {}

        for c in goods_market_participants.keys():
            seller_priorities[c] = np.array(
                [
                    goods_market_participants[c][i].transactor_settings["Buyer Priority"]
                    for i in range(len(goods_market_participants[c]))
                ]
            )
            buyer_priorities[c] = np.array(
                [
                    goods_market_participants[c][i].transactor_settings["Seller Priority"]
                    for i in range(len(goods_market_participants[c]))
                ]
            )

        return cls(
            n_industries,
            functions,
            trade_proportions,
            ts,
            goods_market_participants,
            states=states,
            buyer_priorities=buyer_priorities,
            seller_priorities=seller_priorities,
        )

    def prepare(self, collect_sd: bool = True) -> None:
        # Prepare agents
        self.functions["clearing"].prepare(goods_market_participants=self.goods_market_participants)
        if collect_sd:
            total_supply, total_demand = self.functions["clearing"].collect_all_supply_and_demand(
                goods_market_participants=self.goods_market_participants,
                n_industries=self.n_industries,
            )
            self.ts.total_industry_supply.append(total_supply)
            self.ts.total_industry_demand.append(total_demand)

        # Update the supply chain
        #
        # self.states["previous_supply_chain"] = deepcopy(  # turned off for now
        #     self.states["current_supply_chain"]
        # )
        #
        self.states["current_supply_chain"] = {g: {} for g in range(self.n_industries)}

    def clear(self) -> None:
        self.functions["clearing"].clear(
            goods_market_participants=self.goods_market_participants,
            n_industries=self.n_industries,
            default_origin_trade_proportions=self.states["origin_trade_proportions"],
            default_destin_trade_proportions=self.states["destin_trade_proportions"],
            buyer_priorities=self.buyer_priorities,
            previous_supply_chain=self.states["previous_supply_chain"],
            current_supply_chain=self.states["current_supply_chain"],
        )

    def record(self) -> None:
        self.functions["clearing"].record(goods_market_participants=self.goods_market_participants)

    def save_to_h5(self, h5_file: pd.HDFStore) -> None:
        group = h5_file.create_group("GM")
        self.ts.write_to_h5("GM", group)


def format_array(arr):
    return np.array2string(arr, formatter={"float_kind": lambda x: "{:.2e}".format(x)})
