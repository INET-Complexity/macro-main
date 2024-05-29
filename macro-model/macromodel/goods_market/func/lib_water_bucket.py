import numpy as np

from numba import njit

from macromodel.agents.agent import Agent
from macromodel.goods_market.value_type import ValueType

from typing import Optional, Tuple


@njit
def get_trade_proportions(
    n_countries: int,
    default_origin_trade_proportions: np.ndarray,
    default_destin_trade_proportions: np.ndarray,
    average_prices_by_country: np.ndarray,
    temperature: float,
    real_country_prioritisation: float,
    row_index: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    # Origin trade proportions
    origin_trade_proportions = np.zeros_like(default_origin_trade_proportions)
    for c1 in range(n_countries):
        origin_trade_proportions[c1] = (
            np.exp(-temperature * average_prices_by_country[c1]) * default_origin_trade_proportions[c1]
        )
    """
    origin_trade_proportions[-1] = (
        1 - max(0.0, min(1.0, real_country_prioritisation))
    ) * default_origin_trade_proportions[-1]
    """
    for c2 in range(n_countries):
        origin_trade_proportions[:, c2] /= np.sum(origin_trade_proportions[:, c2], axis=0)

    # Destin trade proportions
    destin_trade_proportions = default_destin_trade_proportions.copy()
    destin_trade_proportions[:, row_index] = (
        1 - max(0.0, min(1.0, real_country_prioritisation))
    ) * default_destin_trade_proportions[:, row_index]
    for c1 in range(n_countries):
        destin_trade_proportions[c1] /= np.sum(destin_trade_proportions[c1], axis=0)

    return origin_trade_proportions, destin_trade_proportions


@njit
def invert_permutation(p: np.ndarray) -> np.ndarray:
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s


@njit
def fill_buckets(
    capacities: np.ndarray,
    fill_amount: float,
    priorities: np.ndarray,
    minimum_fill: float,
) -> np.ndarray:
    if np.sum(capacities) == np.sum(capacities) + 1:
        return np.full_like(capacities, fill_amount / len(capacities))
    if np.sum(capacities) == 0 or fill_amount == 0.0:
        return np.zeros_like(capacities)
    capacities_sorted = capacities[priorities]
    filled_capacities = np.zeros_like(capacities_sorted)
    if minimum_fill > 0.0:
        filled_capacities += np.minimum(
            capacities_sorted,
            capacities_sorted / np.sum(capacities_sorted) * minimum_fill * fill_amount,
        )
    filled_ind = np.where((capacities_sorted - filled_capacities).cumsum() < fill_amount - np.sum(filled_capacities))[0]
    filled_capacities[filled_ind] = capacities_sorted[filled_ind]
    if len(filled_ind) < len(filled_capacities):
        filled_capacities[len(filled_ind)] += fill_amount - np.sum(filled_capacities)
        filled_capacities[len(filled_ind)] = min(
            filled_capacities[len(filled_ind)],
            capacities_sorted[len(filled_ind)],
        )
    return filled_capacities[invert_permutation(priorities)]


# njit no possible
def get_seller_priorities_stochastic(
    productions: np.ndarray,
    prices: np.ndarray,
    price_temperature: float,
    distribution_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if np.sum(productions) == 0.0:
        return np.full(productions.shape[0], 1.0 / productions.shape[0]), np.random.choice(
            productions.shape[0], productions.shape[0], replace=False
        )
    distribution_production = productions / np.sum(productions)
    distribution_prices = np.exp(-price_temperature * prices)
    if np.sum(distribution_prices) == 0.0:
        return np.full(productions.shape[0], 1.0 / productions.shape[0]), np.random.choice(
            productions.shape[0], productions.shape[0], replace=False
        )
    distribution_prices /= np.sum(distribution_prices)
    if distribution_type == "multiplicative":
        distribution = distribution_production * distribution_prices
    elif distribution_type == "additive":
        distribution = 0.5 * (distribution_production + distribution_prices)
    else:
        raise ValueError("Unknown distribution type", distribution_type)
    distribution[distribution == 0.0] = 1e-20
    distribution /= np.sum(distribution)
    return distribution, np.random.choice(len(distribution), len(distribution), replace=False, p=distribution)


@njit
def get_seller_priorities_deterministic(
    productions: np.ndarray,
    prices: np.ndarray,
    price_temperature: float,
    distribution_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if np.sum(productions) == 0.0:
        return np.full(productions.shape[0], 1.0 / productions.shape[0]), np.random.choice(
            productions.shape[0], productions.shape[0], replace=False
        )
    distribution_production = productions / np.sum(productions)
    distribution_prices = np.exp(-price_temperature * prices)
    if np.sum(distribution_prices) == 0.0:
        return np.full(productions.shape[0], 1.0 / productions.shape[0]), np.random.choice(
            productions.shape[0], productions.shape[0], replace=False
        )
    distribution_prices /= np.sum(distribution_prices)
    if distribution_type == "multiplicative":
        distribution = distribution_production * distribution_prices
    elif distribution_type == "additive":
        distribution = 0.5 * (distribution_production + distribution_prices)
    else:
        raise ValueError("Unknown distribution type", distribution_type)
    return distribution, np.argsort(distribution)[::-1]


@njit
def get_buyer_priorities(n_buyers: int) -> np.ndarray:
    return np.random.choice(n_buyers, n_buyers, replace=False)


@njit
def get_transactor_buyer_priorities(priorities: np.ndarray, prioritise: bool) -> np.ndarray:
    if prioritise:
        high_prio, low_prio = (
            np.where(priorities == 1)[0],
            np.where(priorities == 0)[0],
        )
        return np.concatenate(
            (
                np.random.choice(high_prio, len(high_prio), replace=False),
                np.random.choice(low_prio, len(low_prio), replace=False),
            )
        )
    else:
        return np.random.choice(len(priorities), len(priorities), replace=False)


def clear_water_bucket(
    goods_market_participants: dict[str, list[Agent]],
    buyer_priority: dict[str, np.ndarray],
    n_industries: int,
    total_real_supply: dict[str, np.ndarray],
    aggr_real_supply: np.ndarray,
    average_goods_price: np.ndarray,
    total_real_demand: dict[str, np.ndarray],
    aggr_real_demand: np.ndarray,
    price_temperature: float,
    distribution_type: str,
    seller_minimum_fill: float,
    buyer_minimum_fill_macro: float,
    buyer_minimum_fill_micro: float,
    deterministic: bool,
    consider_buyer_priorities: bool,
    sell_high_prio_only: bool = False,
    buy_high_prio_only: bool = False,
    from_country: Optional[int] = None,
    to_country: Optional[int] = None,
    origin_trade_proportions: Optional[np.ndarray] = None,
    destin_trade_proportions: Optional[np.ndarray] = None,
    exclude_row: bool = False,
    with_buyer_value_type: Optional[ValueType] = None,
) -> None:
    if from_country is None:
        from_country_names = list(goods_market_participants.keys())
        if exclude_row:
            if "ROW" in from_country_names:
                from_country_names.remove("ROW")
    else:
        from_country_names = [list(goods_market_participants.keys())[from_country]]
    if to_country is None:
        to_country_names = list(goods_market_participants.keys())
        if exclude_row:
            if "ROW" in to_country_names:
                to_country_names.remove("ROW")
    else:
        to_country_names = [list(goods_market_participants.keys())[to_country]]
    for g in range(n_industries):
        if origin_trade_proportions is None or destin_trade_proportions is None:
            origin_trade_prop = 1.0
            destin_trade_prop = 1.0
        else:
            origin_trade_prop = origin_trade_proportions[g]
            destin_trade_prop = destin_trade_proportions[g]
        if aggr_real_supply[g] == 0 or aggr_real_demand[g] == 0:
            continue
        if aggr_real_supply[g] > aggr_real_demand[g]:
            # Seller
            for country_name in from_country_names:
                for transactor in goods_market_participants[country_name]:
                    if transactor.transactor_seller_states["Priority"] == 1 or not sell_high_prio_only:
                        if transactor.transactor_seller_states["Value Type"] == ValueType.REAL:
                            ind = transactor.transactor_seller_states["Industries"] == g
                            if np.any(transactor.transactor_seller_states["Remaining Goods"][ind] > 0.0):
                                if deterministic:
                                    _, seller_priorities = get_seller_priorities_deterministic(
                                        productions=transactor.transactor_seller_states["Initial Goods"][ind],
                                        prices=transactor.transactor_seller_states["Prices"][ind],
                                        price_temperature=price_temperature,
                                        distribution_type=distribution_type,
                                    )
                                else:
                                    _, seller_priorities = get_seller_priorities_stochastic(
                                        productions=transactor.transactor_seller_states["Initial Goods"][ind],
                                        prices=transactor.transactor_seller_states["Prices"][ind],
                                        price_temperature=price_temperature,
                                        distribution_type=distribution_type,
                                    )
                                real_amount = fill_buckets(
                                    capacities=np.minimum(
                                        destin_trade_prop * transactor.transactor_seller_states["Initial Goods"][ind],
                                        transactor.transactor_seller_states["Remaining Goods"][ind],
                                    ),
                                    fill_amount=total_real_supply[country_name][g]
                                    / aggr_real_supply[g]
                                    * aggr_real_demand[g],
                                    priorities=seller_priorities,
                                    minimum_fill=seller_minimum_fill,
                                )
                                for rec_country in total_real_demand.keys():
                                    transactor.transactor_seller_states["Real Amount sold to " + rec_country][ind] += (
                                        real_amount * total_real_demand[rec_country][g] / aggr_real_demand[g]
                                    )
                                transactor.transactor_seller_states["Real Amount sold"][ind] += real_amount
                                transactor.transactor_seller_states["Remaining Goods"][ind] -= real_amount

            # Buyer
            for country_name in to_country_names:
                for transactor in goods_market_participants[country_name]:
                    if (
                        with_buyer_value_type is None
                        or transactor.transactor_buyer_states["Value Type"] == with_buyer_value_type
                    ):
                        if transactor.transactor_buyer_states["Priority"] == 1 or not buy_high_prio_only:
                            real_prop_rem = np.minimum(
                                origin_trade_prop * transactor.transactor_buyer_states["Initial Goods"][:, g],
                                transactor.transactor_buyer_states["Remaining Goods"][:, g],
                            )
                            if transactor.transactor_buyer_states["Value Type"] == ValueType.NOMINAL:
                                real_prop_rem /= average_goods_price[g]
                            transactor.transactor_buyer_states["Nominal Amount spent"][:, g] += (
                                average_goods_price[g] * real_prop_rem
                            )
                            transactor.transactor_buyer_states["Real Amount bought"][:, g] += real_prop_rem
                            for sell_country in total_real_supply.keys():
                                transactor.transactor_buyer_states[
                                    "Nominal Amount spent on Goods from " + sell_country
                                ][:, g] += (
                                    (average_goods_price[g] * real_prop_rem)
                                    * total_real_supply[sell_country][g]
                                    / aggr_real_supply[g]
                                )
                                transactor.transactor_buyer_states["Real Amount bought from " + sell_country][:, g] += (
                                    real_prop_rem * total_real_supply[sell_country][g] / aggr_real_supply[g]
                                )
                            if transactor.transactor_buyer_states["Value Type"] == ValueType.NOMINAL:
                                transactor.transactor_buyer_states["Remaining Goods"][:, g] -= (
                                    average_goods_price[g] * real_prop_rem
                                )
                            else:
                                transactor.transactor_buyer_states["Remaining Goods"][:, g] -= real_prop_rem
        else:
            # Seller
            for country_name in from_country_names:
                for transactor in goods_market_participants[country_name]:
                    if transactor.transactor_seller_states["Priority"] == 1 or not sell_high_prio_only:
                        if transactor.transactor_seller_states["Value Type"] == ValueType.REAL:
                            ind = transactor.transactor_seller_states["Industries"] == g
                            rem_min = np.minimum(
                                destin_trade_prop * transactor.transactor_seller_states["Initial Goods"][ind],
                                transactor.transactor_seller_states["Remaining Goods"][ind],
                            )
                            transactor.transactor_seller_states["Real Amount sold"][ind] += rem_min
                            for buy_country in total_real_demand.keys():
                                transactor.transactor_seller_states["Real Amount sold to " + buy_country][ind] += (
                                    rem_min / aggr_real_demand[g] * total_real_demand[buy_country][g]
                                )
                            transactor.transactor_seller_states["Remaining Goods"][ind] -= rem_min

            # Buyer
            for country_name in to_country_names:
                # Buyer prioritisation
                transactor_buyer_priorities = get_transactor_buyer_priorities(
                    priorities=buyer_priority[country_name],
                    prioritise=consider_buyer_priorities,
                )
                transactor_real_cap = np.zeros(len(goods_market_participants[country_name]))
                for i in range(len(goods_market_participants[country_name])):
                    rem_amount = np.minimum(
                        origin_trade_prop
                        * goods_market_participants[country_name][i].transactor_buyer_states["Initial Goods"][:, g],
                        goods_market_participants[country_name][i].transactor_buyer_states["Remaining Goods"][:, g],
                    ).sum()
                    if (
                        goods_market_participants[country_name][i].transactor_buyer_states["Value Type"]
                        == ValueType.REAL
                    ):
                        transactor_real_cap[i] = rem_amount
                    else:
                        transactor_real_cap[i] = rem_amount / average_goods_price[g]
                transactor_total_real_supply = fill_buckets(
                    capacities=transactor_real_cap,
                    fill_amount=total_real_demand[country_name][g] / aggr_real_demand[g] * aggr_real_supply[g],
                    priorities=transactor_buyer_priorities,
                    minimum_fill=buyer_minimum_fill_macro,
                )
                if np.sum(np.isnan(transactor_total_real_supply)) > 0:
                    print(average_goods_price[g], transactor_total_real_supply)
                    print(transactor_real_cap)
                    print(total_real_demand[country_name][g] / aggr_real_demand[g] * aggr_real_supply[g])
                    exit()

                # Iterate over buyers
                for i, transactor in enumerate(goods_market_participants[country_name]):
                    prop_real = np.minimum(
                        origin_trade_prop * transactor.transactor_buyer_states["Initial Goods"][:, g],
                        transactor.transactor_buyer_states["Remaining Goods"][:, g],
                    )
                    if transactor.transactor_buyer_states["Value Type"] == ValueType.NOMINAL:
                        prop_real /= average_goods_price[g]
                    buyer_priorities = get_buyer_priorities(
                        n_buyers=transactor.transactor_buyer_states["Remaining Goods"].shape[0]
                    )
                    real_amount_bought = fill_buckets(
                        capacities=prop_real,
                        fill_amount=float(transactor_total_real_supply[i]),
                        priorities=buyer_priorities,
                        minimum_fill=buyer_minimum_fill_micro,
                    )
                    if np.sum(np.isnan(real_amount_bought)) > 0:
                        print(average_goods_price[g], real_amount_bought)
                        print(prop_real)
                        print(float(transactor_total_real_supply[i]))
                        print(type(transactor))
                        exit()
                    for sell_country in total_real_supply.keys():
                        real_amount_bought_by_country = (
                            real_amount_bought * total_real_supply[sell_country][g] / aggr_real_supply[g]
                        )
                        transactor.transactor_buyer_states["Real Amount bought from " + sell_country][
                            :, g
                        ] += real_amount_bought_by_country
                        transactor.transactor_buyer_states["Nominal Amount spent on Goods from " + sell_country][
                            :, g
                        ] += (average_goods_price[g] * real_amount_bought_by_country)
                    if np.isnan(average_goods_price[g]) or np.sum(np.isnan(real_amount_bought)) > 0:
                        print(average_goods_price[g], real_amount_bought)
                        exit()
                    transactor.transactor_buyer_states["Nominal Amount spent"][:, g] += (
                        average_goods_price[g] * real_amount_bought
                    )
                    transactor.transactor_buyer_states["Real Amount bought"][:, g] += real_amount_bought
                    if transactor.transactor_buyer_states["Value Type"] == ValueType.REAL:
                        transactor.transactor_buyer_states["Remaining Goods"][:, g] -= real_amount_bought
                    else:
                        transactor.transactor_buyer_states["Remaining Goods"][:, g] -= (
                            average_goods_price[g] * real_amount_bought
                        )
