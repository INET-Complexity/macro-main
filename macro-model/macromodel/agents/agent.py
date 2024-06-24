from copy import deepcopy

import numpy as np
from typing import Any, Optional

from macromodel.firms.firm_ts import FirmTimeSeries
from macromodel.timeseries import TimeSeries
from numba import njit


class Agent:
    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        n_transactors_sell: int,
        n_transactors_buy: int,
        ts: TimeSeries | FirmTimeSeries,
        states: dict[str, Any],
        transactor_settings: Optional[dict[str, Any]] = None,
    ):
        self.country_name = country_name
        self.all_country_names = all_country_names
        self.n_industries = n_industries
        self.n_transactors_sell = n_transactors_sell
        self.n_transactors_buy = n_transactors_buy
        self.states = states

        self.initial_states = deepcopy(states)

        self.transactor_settings = transactor_settings if transactor_settings else {}

        self.transactor_buyer_states = {}
        self.transactor_seller_states = {}
        self.exchange_rate_usd_to_lcu = None

        # Initiate the time series
        self.ts = ts
        self.initiate_ts()

    def initiate_ts(self) -> None:
        if self.n_transactors_buy > 0 and self.n_transactors_sell > 0:
            self.ts["real_amount_sold"] = np.full(self.n_transactors_sell, np.nan)
            for country_name in self.all_country_names:
                self.ts["real_amount_sold_to_" + country_name] = np.full(self.n_transactors_sell, np.nan)
            self.ts["nominal_amount_sold_in_lcu"] = np.full(self.n_transactors_sell, np.nan)
            for country_name in self.all_country_names:
                self.ts["nominal_amount_sold_in_lcu_to_" + country_name] = np.full(self.n_transactors_sell, np.nan)
            self.ts["real_excess_demand"] = np.full(self.n_transactors_sell, np.nan)

            self.ts["nominal_amount_spent_in_usd"] = np.full((self.n_transactors_buy, self.n_industries), np.nan)
            for country_name in self.all_country_names:
                self.ts["nominal_amount_spent_in_usd_to_" + country_name] = np.full(
                    (self.n_transactors_buy, self.n_industries), np.nan
                )
            self.ts["nominal_amount_spent_in_lcu"] = np.full((self.n_transactors_buy, self.n_industries), np.nan)
            for country_name in self.all_country_names:
                self.ts["nominal_amount_spent_in_lcu_to_" + country_name] = np.full(
                    (self.n_transactors_buy, self.n_industries), np.nan
                )
            self.ts["real_amount_bought"] = np.full((self.n_transactors_buy, self.n_industries), np.nan)
            for country_name in self.all_country_names:
                self.ts["real_amount_bought_from_" + country_name] = np.full(
                    (self.n_transactors_buy, self.n_industries), np.nan
                )

    def __str__(self):
        return f"{self.__class__.__name__}({self.country_name})"

    def set_goods_to_buy(self, buy_init: np.ndarray) -> None:
        self.transactor_buyer_states["Initial Goods"] = buy_init

    def set_goods_to_sell(self, sell_init: np.ndarray) -> None:
        self.transactor_seller_states["Initial Goods"] = sell_init

    def set_maximum_excess_demand(self, max_excess_demand: np.ndarray) -> None:
        self.transactor_seller_states["Remaining Excess Goods"] = max_excess_demand

    def set_prices(self, sell_price: np.ndarray) -> None:
        self.transactor_seller_states["Prices"] = sell_price

    def set_seller_industries(self, industries: np.ndarray) -> None:
        self.transactor_seller_states["Industries"] = industries

    def set_exchange_rate(self, exchange_rate_usd_to_lcu: float) -> None:
        self.exchange_rate_usd_to_lcu = exchange_rate_usd_to_lcu

    def prepare(self) -> None:
        # Value type and priority
        self.transactor_buyer_states["Value Type"] = self.transactor_settings["Buyer Value Type"]
        self.transactor_buyer_states["Priority"] = self.transactor_settings["Buyer Priority"]
        self.transactor_seller_states["Value Type"] = self.transactor_settings["Seller Value Type"]
        self.transactor_seller_states["Priority"] = self.transactor_settings["Seller Priority"]

        # Remaining goods to buy or to sell
        self.transactor_buyer_states["Remaining Goods"] = self.transactor_buyer_states["Initial Goods"].copy()
        self.transactor_seller_states["Remaining Goods"] = self.transactor_seller_states["Initial Goods"].copy()

        # Amount sold
        self.transactor_seller_states["Real Amount sold"] = np.zeros(
            self.transactor_seller_states["Initial Goods"].shape
        )
        for country_name in self.all_country_names:
            self.transactor_seller_states["Real Amount sold to " + country_name] = np.zeros(
                self.transactor_seller_states["Initial Goods"].shape
            )

        # Amount spent
        self.transactor_buyer_states["Nominal Amount spent"] = np.zeros(
            self.transactor_buyer_states["Initial Goods"].shape
        )
        for country_name in self.all_country_names:
            self.transactor_buyer_states["Nominal Amount spent on Goods from " + country_name] = np.zeros(
                self.transactor_buyer_states["Initial Goods"].shape
            )

        # Amount bought
        self.transactor_buyer_states["Real Amount bought"] = np.zeros(
            self.transactor_buyer_states["Initial Goods"].shape
        )
        for country_name in self.all_country_names:
            self.transactor_buyer_states["Real Amount bought from " + country_name] = np.zeros(
                self.transactor_buyer_states["Initial Goods"].shape
            )

        # Excess demand
        self.transactor_seller_states["Real Excess Demand"] = np.zeros(
            self.transactor_seller_states["Initial Goods"].shape
        )

    def record(self) -> None:
        # Round
        self.transactor_seller_states["Real Amount sold"] = round_pos(self.transactor_seller_states["Real Amount sold"])
        for country_name in self.all_country_names:
            self.transactor_seller_states["Real Amount sold to " + country_name] = round_pos(
                self.transactor_seller_states["Real Amount sold to " + country_name]
            )
        self.transactor_seller_states["Real Excess Demand"] = round_pos(
            self.transactor_seller_states["Real Excess Demand"]
        )
        self.transactor_buyer_states["Nominal Amount spent"] = round_pos(
            self.transactor_buyer_states["Nominal Amount spent"]
        )
        for country_name in self.all_country_names:
            self.transactor_buyer_states["Nominal Amount spent on Goods from " + country_name] = round_pos(
                self.transactor_buyer_states["Nominal Amount spent on Goods from " + country_name]
            )
        self.transactor_buyer_states["Real Amount bought"] = round_pos(
            self.transactor_buyer_states["Real Amount bought"]
        )
        for country_name in self.all_country_names:
            self.transactor_buyer_states["Real Amount bought from " + country_name] = round_pos(
                self.transactor_buyer_states["Real Amount bought from " + country_name]
            )

        # Update time series for sellers
        if "Prices" in self.transactor_seller_states.keys():
            self.ts.real_amount_sold.append(self.transactor_seller_states["Real Amount sold"])
            for country_name in self.all_country_names:
                self.ts.dicts["real_amount_sold_to_" + country_name].append(
                    self.transactor_seller_states["Real Amount sold to " + country_name]
                )
            self.ts.nominal_amount_sold_in_lcu.append(
                self.exchange_rate_usd_to_lcu
                * self.transactor_seller_states["Prices"]
                * self.transactor_seller_states["Real Amount sold"]
            )
            for country_name in self.all_country_names:
                self.ts.dicts["nominal_amount_sold_in_lcu_to_" + country_name].append(
                    self.exchange_rate_usd_to_lcu
                    * self.transactor_seller_states["Prices"]
                    * self.transactor_seller_states["Real Amount sold to " + country_name]
                )
            self.ts.real_excess_demand.append(self.transactor_seller_states["Real Excess Demand"])
        else:
            self.ts.real_amount_sold.append(np.full(self.n_transactors_sell, np.nan))
            for country_name in self.all_country_names:
                self.ts.dicts["real_amount_sold_to_" + country_name].append(np.full(self.n_transactors_sell, np.nan))
            self.ts.nominal_amount_sold_in_lcu.append(np.full(self.n_transactors_sell, np.nan))
            for country_name in self.all_country_names:
                self.ts.dicts["nominal_amount_sold_in_lcu_to_" + country_name].append(
                    np.full(self.n_transactors_sell, np.nan)
                )
            self.ts.real_excess_demand.append(np.full(self.n_transactors_sell, np.nan))

        # Update time series for buyers
        self.ts.nominal_amount_spent_in_usd.append(self.transactor_buyer_states["Nominal Amount spent"])
        for country_name in self.all_country_names:
            self.ts.dicts["nominal_amount_spent_in_usd_to_" + country_name].append(
                self.transactor_buyer_states["Nominal Amount spent on Goods from " + country_name]
            )
        self.ts.nominal_amount_spent_in_lcu.append(
            self.exchange_rate_usd_to_lcu * self.transactor_buyer_states["Nominal Amount spent"]
        )
        for country_name in self.all_country_names:
            self.ts.dicts["nominal_amount_spent_in_lcu_to_" + country_name].append(
                self.exchange_rate_usd_to_lcu
                * self.transactor_buyer_states["Nominal Amount spent on Goods from " + country_name]
            )
        self.ts.real_amount_bought.append(self.transactor_buyer_states["Real Amount bought"])
        for country_name in self.all_country_names:
            self.ts.dicts["real_amount_bought_from_" + country_name].append(
                self.transactor_buyer_states["Real Amount bought from " + country_name]
            )

    def gen_reset(self) -> None:
        self.states = deepcopy(self.initial_states)
        self.ts.reset()


@njit
def round_pos(x: np.ndarray, decimals: int = 16) -> np.ndarray:
    r = np.round(x, decimals)
    return np.maximum(0.0, r)
