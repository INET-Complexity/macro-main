from pathlib import Path

from model.timeseries import TimeSeries
from model.util.function_mapping import get_functions
from model.goods_market.goods_market_ts import create_goods_market_timeseries

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
            loc="model.goods_market",
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

    def prepare(self, verbose: bool = False) -> None:
        self.functions["clearing"].prepare()
        total_supply, total_demand = self.functions["clearing"].collect_all_supply_and_demand(verbose=verbose)
        self.ts.total_industry_supply.append(total_supply)
        self.ts.total_industry_demand.append(total_demand)
        if verbose:
            print("TOTALS GM")
            print(total_supply)
            print(total_demand)
            print("---------------------------------")

    def clear(self) -> None:
        self.functions["clearing"].clear()

    def record(self) -> None:
        self.functions["clearing"].record()
