import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional

from inet_macromodel.agents.agent import Agent
from inet_macromodel.goods_market.value_type import ValueType


def get_trade_proportions(
    country_names: list[str],
    default_trade_proportions: pd.DataFrame,
    average_prices: dict[str, np.ndarray],
    temperature: float,
    n_industries: int,
) -> pd.DataFrame:
    new_trade_proportions = {"start_country": [], "end_country": [], "industry": [], "value": []}
    for end_country in country_names:
        for start_country in country_names:
            if start_country == end_country == "ROW":
                continue
            new_trade_proportions["start_country"] += [start_country] * n_industries
            new_trade_proportions["end_country"] += [end_country] * n_industries
            new_trade_proportions["industry"] += list(range(n_industries))
            new_trade_proportions["value"] += list(
                default_trade_proportions.xs(start_country, axis=0, level=0)
                .xs(end_country, axis=0, level=0)
                .values.flatten()
                * np.exp(-temperature * average_prices[start_country])
            )
    new_trade_proportions = pd.DataFrame(new_trade_proportions).set_index(["start_country", "end_country", "industry"])
    new_trade_proportions = new_trade_proportions / new_trade_proportions.groupby(["end_country", "industry"]).sum()
    new_trade_proportions = new_trade_proportions.reorder_levels(["start_country", "end_country", "industry"])
    return new_trade_proportions


def get_split_sum(val: np.ndarray, groups: np.ndarray, n_industries) -> np.ndarray:
    split_sum = np.zeros(n_industries)
    for _g in range(n_industries):
        split_sum[_g] = val[groups == _g].sum()
    return split_sum


def collect_seller_info(
    goods_market_participants: dict[str, list[Agent]],
    n_industries: int,
    high_prio_only: bool = False,
    from_country: Optional[str] = None,
    exclude_row: bool = False,
    field: str = "Remaining Goods",
) -> Tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if from_country is None:
        country_names = list(goods_market_participants.keys())
        if exclude_row:
            if "ROW" in country_names:
                country_names.remove("ROW")
    else:
        country_names = [from_country]
    total_real_supply = {c: np.zeros(n_industries) for c in country_names}
    total_nominal_supply = {c: np.zeros(n_industries) for c in country_names}
    for country_name in country_names:
        for transactor in goods_market_participants[country_name]:
            if transactor.transactor_seller_states["Priority"] == 1 or not high_prio_only:
                if transactor.transactor_seller_states["Value Type"] == ValueType.REAL:
                    total_real_supply[country_name] += get_split_sum(
                        transactor.transactor_seller_states[field],
                        transactor.transactor_seller_states["Industries"],
                        n_industries,
                    )
                    total_nominal_supply[country_name] += get_split_sum(
                        transactor.transactor_seller_states[field] * transactor.transactor_seller_states["Prices"],
                        transactor.transactor_seller_states["Industries"],
                        n_industries,
                    )
                    logging.debug("\n")
                    logging.debug("Seller information for %s", country_name)
                    logging.debug("Transactor %s", transactor)
                    logging.debug(
                        "Total %s: %.2e",
                        field,
                        np.sum(
                            get_split_sum(
                                transactor.transactor_seller_states[field]
                                * transactor.transactor_seller_states["Prices"],
                                transactor.transactor_seller_states["Industries"],
                                n_industries,
                            )
                        ),
                    )

    # Calculate the average price
    aggr_real_supply = np.stack(list(total_real_supply.values()), axis=0).sum(axis=0)
    aggr_nominal_supply = np.stack(list(total_nominal_supply.values()), axis=0).sum(axis=0)
    price = np.divide(
        aggr_nominal_supply,
        aggr_real_supply,
        out=np.zeros(n_industries),
        where=aggr_real_supply != 0,
    )

    return (
        total_real_supply,
        aggr_real_supply,
        total_nominal_supply,
        aggr_nominal_supply,
        price,
    )


def collect_buyer_info(
    goods_market_participants: dict[str, list[Agent]],
    average_price: np.ndarray,
    n_industries: int,
    high_prio_only: bool = False,
    to_country: Optional[str] = None,
    trade_proportions: Optional[np.ndarray] = None,
    exclude_row: bool = False,
    with_value_type: Optional[ValueType] = None,
) -> Tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray]:
    if to_country is None:
        country_names = list(goods_market_participants.keys())
        if exclude_row:
            if "ROW" in country_names:
                country_names.remove("ROW")
    else:
        country_names = [to_country]
    total_real_demand = {c: np.zeros(n_industries) for c in country_names}
    total_nominal_demand = {c: np.zeros(n_industries) for c in country_names}
    for country_name in country_names:
        for transactor in goods_market_participants[country_name]:
            if with_value_type is None or transactor.transactor_buyer_states["Value Type"] == with_value_type:
                if transactor.transactor_buyer_states["Priority"] == 1 or not high_prio_only:
                    if transactor.transactor_buyer_states["Value Type"] == ValueType.REAL:
                        total_real_demand[country_name] += transactor.transactor_buyer_states["Remaining Goods"].sum(
                            axis=0
                        )
                        total_nominal_demand[country_name] += (
                            transactor.transactor_buyer_states["Remaining Goods"].sum(axis=0) * average_price
                        )
                    elif transactor.transactor_buyer_states["Value Type"] == ValueType.NOMINAL:
                        total_real_demand[country_name] += np.divide(
                            transactor.transactor_buyer_states["Remaining Goods"].sum(axis=0),
                            average_price,
                            out=np.full(
                                transactor.transactor_buyer_states["Remaining Goods"].shape[1],
                                np.nan,
                            ),
                            where=average_price != 0,
                        )
                        total_nominal_demand[country_name] += transactor.transactor_buyer_states["Remaining Goods"].sum(
                            axis=0
                        )
                    logging.debug("\n")
                    logging.debug("Buyer information for %s", country_name)
                    logging.debug("Transactor %s", transactor)
                    logging.debug(
                        "Total remaining goods: %.2e", np.sum(transactor.transactor_buyer_states["Remaining Goods"])
                    )

    # Calculate sums
    aggr_real_demand = np.stack(list(total_real_demand.values()), axis=0).sum(axis=0)
    aggr_nominal_demand = np.stack(list(total_nominal_demand.values()), axis=0).sum(axis=0)

    # Scale
    if trade_proportions is not None:
        aggr_real_demand *= trade_proportions
        aggr_nominal_demand * trade_proportions
        for c in total_real_demand.keys():
            total_real_demand[c] *= trade_proportions
        for c in total_nominal_demand.keys():
            total_nominal_demand[c] *= trade_proportions

    return total_real_demand, aggr_real_demand, total_nominal_demand, aggr_nominal_demand


def clear(
    goods_market_participants: dict[str, list[Agent]],
    n_industries: int,
    total_real_supply: dict[str, np.ndarray],
    aggr_real_supply: np.ndarray,
    average_goods_price: np.ndarray,
    total_real_demand: dict[str, np.ndarray],
    aggr_real_demand: np.ndarray,
    sell_high_prio_only: bool,
    buy_high_prio_only: bool,
    from_country: Optional[str] = None,
    to_country: Optional[str] = None,
    trade_proportions: Optional[np.ndarray] = None,
    exclude_row: bool = False,
    with_buyer_value_type: Optional[ValueType] = None,
) -> None:
    if from_country is None:
        from_country_names = list(goods_market_participants.keys())
        if exclude_row:
            if "ROW" in from_country_names:
                from_country_names.remove("ROW")
    else:
        from_country_names = [from_country]
    if to_country is None:
        to_country_names = list(goods_market_participants.keys())
        if exclude_row:
            if "ROW" in to_country_names:
                to_country_names.remove("ROW")
    else:
        to_country_names = [to_country]
    for g in range(n_industries):
        if trade_proportions is None:
            trade_prop = 1.0
        else:
            trade_prop = trade_proportions[g]
        if aggr_real_supply[g] == 0 or aggr_real_demand[g] == 0:
            continue
        if aggr_real_supply[g] > aggr_real_demand[g]:
            # Seller
            for country_name in from_country_names:
                for transactor in goods_market_participants[country_name]:
                    if transactor.transactor_seller_states["Priority"] == 1 or not sell_high_prio_only:
                        if not np.all(transactor.transactor_seller_states["Remaining Goods"] == 0):
                            if transactor.transactor_seller_states["Value Type"] == ValueType.REAL:
                                for rec_country in total_real_demand.keys():
                                    real_amount = (
                                        transactor.transactor_seller_states["Remaining Goods"][
                                            transactor.transactor_seller_states["Industries"] == g
                                        ]
                                        / aggr_real_supply[g]
                                        * total_real_demand[rec_country][g]
                                    )
                                    transactor.transactor_seller_states["Real Amount sold to " + rec_country][
                                        transactor.transactor_seller_states["Industries"] == g
                                    ] += real_amount
                                total_real_amount = (
                                    transactor.transactor_seller_states["Remaining Goods"][
                                        transactor.transactor_seller_states["Industries"] == g
                                    ]
                                    / aggr_real_supply[g]
                                    * aggr_real_demand[g]
                                )
                                transactor.transactor_seller_states["Real Amount sold"][
                                    transactor.transactor_seller_states["Industries"] == g
                                ] += total_real_amount
                                transactor.transactor_seller_states["Remaining Goods"][
                                    transactor.transactor_seller_states["Industries"] == g
                                ] -= (
                                    transactor.transactor_seller_states["Remaining Goods"][
                                        transactor.transactor_seller_states["Industries"] == g
                                    ]
                                    / aggr_real_supply[g]
                                    * aggr_real_demand[g]
                                )
            # Buyer
            for country_name in to_country_names:
                for transactor in goods_market_participants[country_name]:
                    if (
                        with_buyer_value_type is None
                        or transactor.transactor_buyer_states["Value Type"] == with_buyer_value_type
                    ):
                        if transactor.transactor_buyer_states["Priority"] == 1 or not buy_high_prio_only:
                            if transactor.transactor_buyer_states["Value Type"] == ValueType.REAL:
                                transactor.transactor_buyer_states["Nominal Amount spent"][:, g] += (
                                    average_goods_price[g]
                                    * trade_prop
                                    * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                )
                                for sell_country in total_real_supply.keys():
                                    transactor.transactor_buyer_states[
                                        "Nominal Amount spent on Goods from " + sell_country
                                    ][:, g] += (
                                        (
                                            average_goods_price[g]
                                            * trade_prop
                                            * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                        )
                                        * total_real_supply[sell_country][g]
                                        / aggr_real_supply[g]
                                    )
                                transactor.transactor_buyer_states["Real Amount bought"][:, g] += (
                                    trade_prop * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                )
                                for sell_country in total_real_supply.keys():
                                    transactor.transactor_buyer_states["Real Amount bought from " + sell_country][
                                        :, g
                                    ] += (
                                        trade_prop
                                        * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                        * total_real_supply[sell_country][g]
                                        / aggr_real_supply[g]
                                    )
                                transactor.transactor_buyer_states["Remaining Goods"][:, g] -= (
                                    trade_prop * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                )
                            elif transactor.transactor_buyer_states["Value Type"] == ValueType.NOMINAL:
                                transactor.transactor_buyer_states["Nominal Amount spent"][:, g] += (
                                    trade_prop * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                )
                                for sell_country in total_real_supply.keys():
                                    transactor.transactor_buyer_states[
                                        "Nominal Amount spent on Goods from " + sell_country
                                    ][:, g] += (
                                        trade_prop
                                        * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                        * total_real_supply[sell_country][g]
                                        / aggr_real_supply[g]
                                    )
                                transactor.transactor_buyer_states["Real Amount bought"][:, g] += (
                                    trade_prop
                                    * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                    / average_goods_price[g]
                                )
                                for sell_country in total_real_supply.keys():
                                    transactor.transactor_buyer_states["Real Amount bought from " + sell_country][
                                        :, g
                                    ] += (
                                        (
                                            trade_prop
                                            * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                            / average_goods_price[g]
                                        )
                                        * total_real_supply[sell_country][g]
                                        / aggr_real_supply[g]
                                    )
                                transactor.transactor_buyer_states["Remaining Goods"][:, g] -= (
                                    trade_prop * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                )
        else:
            # Seller
            for country_name in from_country_names:
                for transactor in goods_market_participants[country_name]:
                    if transactor.transactor_seller_states["Priority"] == 1 or not sell_high_prio_only:
                        if transactor.transactor_seller_states["Value Type"] == ValueType.REAL:
                            transactor.transactor_seller_states["Real Amount sold"][
                                transactor.transactor_seller_states["Industries"] == g
                            ] += transactor.transactor_seller_states["Remaining Goods"][
                                transactor.transactor_seller_states["Industries"] == g
                            ]
                            for buy_country in total_real_demand.keys():
                                transactor.transactor_seller_states["Real Amount sold to " + buy_country][
                                    transactor.transactor_seller_states["Industries"] == g
                                ] += (
                                    transactor.transactor_seller_states["Remaining Goods"][
                                        transactor.transactor_seller_states["Industries"] == g
                                    ]
                                    / aggr_real_demand[g]
                                    * total_real_demand[buy_country][g]
                                )

                            # Sellers are happy
                            transactor.transactor_seller_states["Remaining Goods"][
                                transactor.transactor_seller_states["Industries"] == g
                            ] = 0.0

            # Buyer
            for country_name in to_country_names:
                for transactor in goods_market_participants[country_name]:
                    if (
                        with_buyer_value_type is None
                        or transactor.transactor_buyer_states["Value Type"] == with_buyer_value_type
                    ):
                        if transactor.transactor_buyer_states["Priority"] == 1 or not buy_high_prio_only:
                            if transactor.transactor_buyer_states["Value Type"] == ValueType.REAL:
                                transactor.transactor_buyer_states["Nominal Amount spent"][:, g] += (
                                    average_goods_price[g]
                                    * trade_prop
                                    * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                    / aggr_real_demand[g]
                                    * aggr_real_supply[g]
                                )
                                for sell_country in total_real_supply.keys():
                                    transactor.transactor_buyer_states[
                                        "Nominal Amount spent on Goods from " + sell_country
                                    ][:, g] += (
                                        average_goods_price[g]
                                        * trade_prop
                                        * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                        / aggr_real_demand[g]
                                        * total_real_supply[sell_country][g]
                                    )
                                transactor.transactor_buyer_states["Real Amount bought"][:, g] += (
                                    trade_prop
                                    * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                    / aggr_real_demand[g]
                                    * aggr_real_supply[g]
                                )
                                for sell_country in total_real_supply.keys():
                                    transactor.transactor_buyer_states["Real Amount bought from " + sell_country][
                                        :, g
                                    ] += (
                                        trade_prop
                                        * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                        / aggr_real_demand[g]
                                        * total_real_supply[sell_country][g]
                                    )
                            elif transactor.transactor_buyer_states["Value Type"] == ValueType.NOMINAL:
                                transactor.transactor_buyer_states["Nominal Amount spent"][:, g] += (
                                    trade_prop
                                    * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                    / aggr_real_demand[g]
                                    * aggr_real_supply[g]
                                )
                                for sell_country in total_real_supply.keys():
                                    transactor.transactor_buyer_states[
                                        "Nominal Amount spent on Goods from " + sell_country
                                    ][:, g] += (
                                        trade_prop
                                        * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                        / aggr_real_demand[g]
                                        * total_real_supply[sell_country][g]
                                    )
                                transactor.transactor_buyer_states["Real Amount bought"][:, g] += (
                                    trade_prop
                                    * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                    / average_goods_price[g]
                                    / aggr_real_demand[g]
                                    * aggr_real_supply[g]
                                )
                                for sell_country in total_real_supply.keys():
                                    transactor.transactor_buyer_states["Real Amount bought from " + sell_country][
                                        :, g
                                    ] += (
                                        trade_prop
                                        * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                        / average_goods_price[g]
                                        / aggr_real_demand[g]
                                        * total_real_supply[sell_country][g]
                                    )

                            # Remaining goods
                            transactor.transactor_buyer_states["Remaining Goods"][:, g] -= (
                                trade_prop
                                * transactor.transactor_buyer_states["Remaining Goods"][:, g]
                                / aggr_real_demand[g]
                                * aggr_real_supply[g]
                            )


def distribute_excess_demand(
    goods_market_participants: dict[str, list[Agent]],
    n_industries: int,
) -> None:
    # Collect initial values
    _, _, _, aggr_nominal_supply, average_price = collect_seller_info(
        goods_market_participants=goods_market_participants,
        n_industries=n_industries,
        high_prio_only=True,
        exclude_row=True,
        field="Initial Goods",
    )
    _, excess_real_demand, _, _ = collect_buyer_info(
        goods_market_participants=goods_market_participants,
        average_price=average_price,
        n_industries=n_industries,
        high_prio_only=False,
    )

    # Distribute excess demand
    for g in range(n_industries):
        if aggr_nominal_supply[g] == 0:
            continue
        for country_name in goods_market_participants.keys():
            if country_name == "ROW":
                continue
            for transactor in goods_market_participants[country_name]:
                if transactor.transactor_seller_states["Priority"] == 1:
                    if transactor.transactor_seller_states["Value Type"] == ValueType.REAL:
                        transactor.transactor_seller_states["Real Excess Demand"][
                            transactor.transactor_seller_states["Industries"] == g
                        ] = (
                            (
                                transactor.transactor_seller_states["Initial Goods"][
                                    transactor.transactor_seller_states["Industries"] == g
                                ]
                                * transactor.transactor_seller_states["Prices"][
                                    transactor.transactor_seller_states["Industries"] == g
                                ]
                            )
                            / aggr_nominal_supply[g]
                            * excess_real_demand[g]
                        )
