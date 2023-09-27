import numpy as np

from model.timeseries import TimeSeries

from typing import Any, Optional


class Agent:
    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        year: int,
        t_max: int,
        n_industries: int,
        n_transactors_sell: int,
        n_transactors_buy: int,
        functions: dict[str, Any],
        parameters: dict[str, Any],
        ts: TimeSeries,
        states: dict[str, Any],
        transactor_settings: Optional[dict[str, Any]] = None,
    ):
        self.country_name = country_name
        self.all_country_names = all_country_names
        self.year = year
        self.t_max = t_max
        self.n_industries = n_industries
        self.n_transactors_sell = n_transactors_sell
        self.n_transactors_buy = n_transactors_buy
        self.functions = functions
        self.parameters = parameters
        self.states = states
        self.transactor_settings = transactor_settings

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

    def set_goods_to_buy(self, buy_init: np.ndarray) -> None:
        self.transactor_buyer_states["Initial Goods"] = buy_init

    def set_goods_to_sell(self, sell_init: np.ndarray) -> None:
        self.transactor_seller_states["Initial Goods"] = sell_init

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
        self.transactor_seller_states["Real Amount sold"] = np.zeros_like(
            self.transactor_seller_states["Initial Goods"]
        )
        for country_name in self.all_country_names:
            self.transactor_seller_states["Real Amount sold to " + country_name] = np.zeros_like(
                self.transactor_seller_states["Initial Goods"]
            )

        # Amount spent
        self.transactor_buyer_states["Nominal Amount spent"] = np.zeros_like(
            self.transactor_buyer_states["Initial Goods"]
        )
        for country_name in self.all_country_names:
            self.transactor_buyer_states["Nominal Amount spent on Goods from " + country_name] = np.zeros_like(
                self.transactor_buyer_states["Initial Goods"]
            )

        # Amount bought
        self.transactor_buyer_states["Real Amount bought"] = np.zeros_like(
            self.transactor_buyer_states["Initial Goods"]
        )
        for country_name in self.all_country_names:
            self.transactor_buyer_states["Real Amount bought from " + country_name] = np.zeros_like(
                self.transactor_buyer_states["Initial Goods"]
            )

        # Real excess demand
        self.transactor_seller_states["Real Excess Demand"] = np.zeros_like(
            self.transactor_seller_states["Initial Goods"]
        )

    def record(self, decimals: int = 4) -> None:
        def round_pos(x: np.ndarray) -> np.ndarray:
            r = np.round(x, decimals)
            r[r < 0] = 0.0
            return r

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
