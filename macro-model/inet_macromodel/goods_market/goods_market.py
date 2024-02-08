import logging
from typing import Any

import numpy as np
import pandas as pd

from agents.agent import Agent
from configurations import GoodsMarketConfiguration
from inet_macromodel.goods_market.goods_market_ts import create_goods_market_timeseries
from inet_macromodel.timeseries import TimeSeries
from inet_macromodel.util.function_mapping import functions_from_model


class GoodsMarket:
    def __init__(
        self,
        n_industries: int,
        functions: dict[str, Any],
        trade_proportions: pd.DataFrame,
        ts: TimeSeries,
    ):
        self.n_industries = n_industries
        self.functions = functions
        self.trade_proportions = trade_proportions.fillna(0.0)
        self.ts = ts

    @classmethod
    def from_data(
        cls,
        n_industries: int,
        trade_proportions: pd.DataFrame,
        configuration: GoodsMarketConfiguration,
        goods_market_participants: dict[str, list[Agent]],
    ) -> "GoodsMarket":
        # Get corresponding functions
        functions = functions_from_model(configuration.functions, loc="inet_macromodel.goods_market")

        functions["clearing"].initiate_agents(
            n_industries=n_industries,
            goods_market_participants=goods_market_participants,
        )
        functions["clearing"].initiate_the_supply_chain(
            initial_supply_chain=None,
        )

        # Create the corresponding time series object
        ts = create_goods_market_timeseries(n_industries)

        return cls(
            n_industries,
            functions,
            trade_proportions,
            ts,
        )

    def prepare(self) -> None:
        self.functions["clearing"].prepare()
        total_supply, total_demand = self.functions["clearing"].collect_all_supply_and_demand()
        self.ts.total_industry_supply.append(total_supply)
        self.ts.total_industry_demand.append(total_demand)
        logging.debug("Total goods market")
        logging.debug(f"Total supply: {format_array(total_supply)}")
        logging.debug(f"Total demand: {format_array(total_demand)}")
        logging.debug("\n")

    def clear(self) -> None:
        self.functions["clearing"].clear(default_trade_proportions=self.trade_proportions)

    def record(self) -> None:
        self.functions["clearing"].record()

    def save_to_h5(self, h5_file: pd.HDFStore) -> None:
        group = h5_file.create_group("GM")
        self.ts.write_to_h5("GM", group)


def format_array(arr):
    return np.array2string(arr, formatter={"float_kind": lambda x: "{:.2e}".format(x)})
