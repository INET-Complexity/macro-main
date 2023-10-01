from pathlib import Path
import logging

import numpy as np

from inet_macromodel.timeseries import TimeSeries
from inet_macromodel.util.function_mapping import get_functions
from inet_macromodel.goods_market.goods_market_ts import create_goods_market_timeseries

from typing import Any


class GoodsMarket:
    def __init__(
        self,
        year: int,
        t_max: int,
        n_industries: int,
        functions: dict[str, Any],
        parameters: dict[str, Any],
        ts: TimeSeries,
    ):
        self.year = year
        self.t_max = t_max
        self.n_industries = n_industries
        self.functions = functions
        self.parameters = parameters
        self.ts = ts

    @classmethod
    def from_data(
        cls,
        year: int,
        t_max: int,
        n_industries: int,
        config: dict[str, Any],
    ) -> "GoodsMarket":
        # Get corresponding functions
        functions = get_functions(
            config["functions"],
            loc="inet_macromodel.goods_market",
            func_dir=Path(__file__).parent / "func",
        )

        # Get corresponding parameters
        if "parameters" in config.keys():
            parameters = config["parameters"].copy()
        else:
            parameters = {}

        # Create the corresponding time series object
        ts = create_goods_market_timeseries(n_industries)

        return cls(
            year,
            t_max,
            n_industries,
            functions,
            parameters,
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
        self.functions["clearing"].clear()

    def record(self) -> None:
        self.functions["clearing"].record()


def format_array(arr):
    return np.array2string(arr, formatter={"float_kind": lambda x: "{:.2e}".format(x)})
