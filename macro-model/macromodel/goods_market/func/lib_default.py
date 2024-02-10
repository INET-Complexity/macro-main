import numpy as np
from typing import Tuple, Optional

from macromodel.agents.agent import Agent
from macromodel.goods_market.value_type import ValueType


def get_split_sum(val: np.ndarray, groups: np.ndarray, n_industries) -> np.ndarray:
    split_sum = np.zeros(n_industries)
    for _g in range(n_industries):
        split_sum[_g] = val[groups == _g].sum()
    return split_sum


def check_sellers_left(
    goods_market_participants: dict[str, list[Agent]],
    n_industries: int,
    field: str,
) -> np.ndarray:
    total_vals = np.zeros(n_industries)
    for country_name in goods_market_participants.keys():
        for transactor in goods_market_participants[country_name]:
            if transactor.transactor_seller_states["Value Type"] != ValueType.NONE:
                total_vals += get_split_sum(
                    transactor.transactor_seller_states[field],
                    transactor.transactor_seller_states["Industries"],
                    n_industries,
                )
    return total_vals > 0.0


def check_buyers_left(
    goods_market_participants: dict[str, list[Agent]],
    n_industries: int,
    field: str,
) -> np.ndarray:
    total_vals = np.zeros(n_industries)
    for country_name in goods_market_participants.keys():
        for transactor in goods_market_participants[country_name]:
            if transactor.transactor_buyer_states["Value Type"] != ValueType.NONE:
                total_vals += transactor.transactor_buyer_states[field].sum(axis=0)
    return total_vals > 0.0


def get_random_buyer_country(
    industry: int,
    goods_market_participants: dict[str, list[Agent]],
    prio_real_countries: bool,
    field: str,
) -> str:
    n_buyers_by_country = {
        c: np.sum(
            [
                np.sum(goods_market_participants[c][i].transactor_buyer_states[field][:, industry] > 0.0)
                for i in range(len(goods_market_participants[c]))
            ]
        )
        for c in goods_market_participants.keys()
        if c != "ROW"
    }
    n_buyers_keys = list(n_buyers_by_country.keys())
    n_buyers_vals = list(n_buyers_by_country.values())
    if prio_real_countries and np.sum(n_buyers_vals) > 0:
        return np.random.choice(n_buyers_keys, p=np.array(n_buyers_vals) / np.sum(n_buyers_vals))

    # If the ROW is considered
    if np.sum(n_buyers_vals) == 0:
        return "ROW"
    total_buying_real_countries = np.sum(
        [
            [
                goods_market_participants[c][i].transactor_buyer_states[field][:, industry].sum()
                for i in range(len(goods_market_participants[c]))
            ]
            for c in goods_market_participants.keys()
            if c != "ROW"
        ]
    )
    total_buying_row = goods_market_participants["ROW"][0].transactor_buyer_states[field][:, industry].sum()
    if total_buying_real_countries == 0:
        n_buyers_by_country["ROW"] = 1
    else:
        n_buyers_by_country["ROW"] = int(np.sum(n_buyers_vals) * total_buying_row / total_buying_real_countries)
    n_buyers_keys = list(n_buyers_by_country.keys())
    n_buyers_vals = list(n_buyers_by_country.values())

    return np.random.choice(n_buyers_keys, p=np.array(n_buyers_vals) / np.sum(n_buyers_vals))


def get_random_buyer_type(
    industry: int,
    country_goods_market_participants: list[Agent],
    prio_high_prio: bool,
    field: str,
) -> Agent:
    if prio_high_prio:
        n_buyers_by_agent = {
            agent: np.sum(agent.transactor_buyer_states[field][:, industry] > 0.0)
            for agent in country_goods_market_participants
            if agent.transactor_buyer_states["Priority"] == 1
        }
        n_buyers_keys = list(n_buyers_by_agent.keys())
        n_buyers_vals = list(n_buyers_by_agent.values())
        if np.sum(n_buyers_vals) > 0:
            return np.random.choice(n_buyers_keys, p=np.array(n_buyers_vals) / np.sum(n_buyers_vals))

    # Considering all agents
    n_buyers_by_agent = {
        agent: np.sum(agent.transactor_buyer_states[field][:, industry] > 0.0)
        for agent in country_goods_market_participants
    }
    n_buyers_keys = list(n_buyers_by_agent.keys())
    n_buyers_vals = list(n_buyers_by_agent.values())
    return np.random.choice(n_buyers_keys, p=np.array(n_buyers_vals) / np.sum(n_buyers_vals))


def get_random_buyer(
    industry: int,
    goods_market_participants: dict[str, list[Agent]],
    prio_real_countries: bool,
    prio_high_prio: bool,
    field: str,
) -> Tuple[Agent, int]:
    # Get the buyers country
    country = get_random_buyer_country(
        industry=industry,
        goods_market_participants=goods_market_participants,
        prio_real_countries=prio_real_countries,
        field=field,
    )

    # Get the buyer type
    buyer = get_random_buyer_type(
        industry=industry,
        country_goods_market_participants=goods_market_participants[country],
        prio_high_prio=prio_high_prio,
        field=field,
    )

    # Choose a random buyer
    buyer_ind = np.random.choice(np.where(buyer.transactor_buyer_states[field][:, industry] > 0.0)[0])

    return buyer, buyer_ind


def pick_previous_seller(
    industry: int,
    chosen_buyer: Agent,
    chosen_buyer_ind: int,
    previous_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]],
    prio_real_countries: bool,
    prio_high_prio: bool,
    prio_domestic_sellers: bool,
    probability_keeping_previous_seller: float,
    field: str,
) -> Optional[Tuple[Agent, int]]:
    if np.random.random() < probability_keeping_previous_seller:
        if (
            chosen_buyer in previous_supply_chain[industry].keys()
            and chosen_buyer_ind in previous_supply_chain[industry][chosen_buyer].keys()
        ):
            possible_sc_sellers, possible_sc_sellers_ind = [], []
            for sc_seller, sc_seller_ind in previous_supply_chain[industry][chosen_buyer][chosen_buyer_ind]:
                if (
                    sc_seller.transactor_seller_states[field][sc_seller_ind] > 0
                    and (not prio_real_countries or sc_seller.country_name != "ROW")
                    and (not prio_high_prio or sc_seller.transactor_seller_states["Priority"] == 1)
                    and (not prio_domestic_sellers or chosen_buyer.country_name == sc_seller.country_name)
                ):
                    possible_sc_sellers.append(sc_seller)
                    possible_sc_sellers_ind.append(sc_seller_ind)
            if len(possible_sc_sellers) > 0:
                random_ind = np.random.choice(len(possible_sc_sellers))
                return (
                    possible_sc_sellers[random_ind],
                    possible_sc_sellers_ind[random_ind],
                )

    return None


def get_random_seller_based_on_distribution(
    industry: int,
    chosen_goods_market_participants: list[Agent],
    price_temperature: float,
    field: str,
) -> Optional[Tuple[Agent, int]]:
    if len(chosen_goods_market_participants) == 0:
        return None

    # Compile all the data
    agent_ls, agent_ind_ls = [], []
    agent_prod_ls, agent_price_ls = [], []
    for agent in chosen_goods_market_participants:
        valid_agent_ind = np.where(
            np.logical_and(
                agent.transactor_seller_states[field] > 0,
                agent.transactor_seller_states["Industries"] == industry,
            )
        )[0]
        agent_ls.extend([agent] * len(valid_agent_ind))
        agent_ind_ls.extend(valid_agent_ind)
        agent_prod_ls.extend(agent.transactor_seller_states["Initial Goods"][valid_agent_ind])
        agent_price_ls.extend(agent.transactor_seller_states["Prices"][valid_agent_ind])
    if len(agent_prod_ls) == 0:
        return None
    agent_prod_ls = np.array(agent_prod_ls)
    agent_price_ls = np.array(agent_price_ls)

    # Build a distribution
    distribution_production = agent_prod_ls / np.sum(agent_prod_ls)
    distribution_prices = np.exp(-price_temperature * agent_price_ls)
    distribution_prices /= np.sum(distribution_prices)
    distribution = distribution_production * distribution_prices
    distribution /= np.sum(distribution)

    # Draw a random seller
    random_ind = np.random.choice(len(distribution), p=distribution)
    return agent_ls[random_ind], agent_ind_ls[random_ind]


def get_random_seller(
    industry: int,
    goods_market_participants: dict[str, list[Agent]],
    chosen_buyer: Agent,
    chosen_buyer_ind: int,
    previous_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]],
    prio_real_countries: bool,
    prio_high_prio_sellers: bool,
    prio_domestic_sellers: bool,
    probability_keeping_previous_seller: float,
    price_temperature: float,
    field: str,
) -> Tuple[Agent, int]:
    # Pick a previous seller
    sc_seller = pick_previous_seller(
        industry=industry,
        chosen_buyer=chosen_buyer,
        chosen_buyer_ind=chosen_buyer_ind,
        previous_supply_chain=previous_supply_chain,
        prio_real_countries=prio_real_countries,
        prio_high_prio=prio_high_prio_sellers,
        prio_domestic_sellers=prio_domestic_sellers,
        probability_keeping_previous_seller=probability_keeping_previous_seller,
        field=field,
    )
    if sc_seller is not None:
        return sc_seller

    # Try to stay domestic
    if prio_domestic_sellers and chosen_buyer.country_name != "ROW":
        # Prioritise high-priority sellers
        if prio_high_prio_sellers:
            chosen_goods_market_participants = []
            for seller in goods_market_participants[chosen_buyer.country_name]:
                if (
                    seller.transactor_seller_states["Value Type"] != ValueType.NONE
                    and seller.transactor_seller_states["Priority"] == 1
                ):
                    chosen_goods_market_participants += [seller]
            chosen_seller = get_random_seller_based_on_distribution(
                industry=industry,
                chosen_goods_market_participants=chosen_goods_market_participants,
                price_temperature=price_temperature,
                field=field,
            )
            if chosen_seller is not None:
                return chosen_seller

        # No consideration of priority status
        chosen_goods_market_participants = []
        for seller in goods_market_participants[chosen_buyer.country_name]:
            if seller.transactor_seller_states["Value Type"] != ValueType.NONE:
                chosen_goods_market_participants += [seller]
        chosen_seller = get_random_seller_based_on_distribution(
            industry=industry,
            chosen_goods_market_participants=chosen_goods_market_participants,
            price_temperature=price_temperature,
            field=field,
        )
        if chosen_seller is not None:
            return chosen_seller

    # Try to prioritise real countries
    if prio_real_countries or chosen_buyer.country_name == "ROW":
        # Prioritise high-priority sellers
        if prio_high_prio_sellers:
            chosen_goods_market_participants = []
            for country_name in goods_market_participants.keys():
                if country_name == "ROW":
                    continue
                for seller in goods_market_participants[country_name]:
                    if (
                        seller.transactor_seller_states["Value Type"] != ValueType.NONE
                        and seller.transactor_seller_states["Priority"] == 1
                    ):
                        chosen_goods_market_participants += [seller]
                chosen_seller = get_random_seller_based_on_distribution(
                    industry=industry,
                    chosen_goods_market_participants=chosen_goods_market_participants,
                    price_temperature=price_temperature,
                    field=field,
                )
                if chosen_seller is not None:
                    return chosen_seller

        # No consideration of priority status
        chosen_goods_market_participants = []
        for country_name in goods_market_participants.keys():
            if country_name == "ROW":
                continue
            for seller in goods_market_participants[country_name]:
                if seller.transactor_seller_states["Value Type"] != ValueType.NONE:
                    chosen_goods_market_participants += [seller]
            chosen_seller = get_random_seller_based_on_distribution(
                industry=industry,
                chosen_goods_market_participants=chosen_goods_market_participants,
                price_temperature=price_temperature,
                field=field,
            )
            if chosen_seller is not None:
                return chosen_seller

    # Choose among all sellers
    chosen_goods_market_participants = []
    for country_name in goods_market_participants.keys():
        for seller in goods_market_participants[country_name]:
            if seller.transactor_seller_states["Value Type"] != ValueType.NONE:
                chosen_goods_market_participants += [seller]
        chosen_seller = get_random_seller_based_on_distribution(
            industry=industry,
            chosen_goods_market_participants=chosen_goods_market_participants,
            price_temperature=price_temperature,
            field=field,
        )
        if chosen_seller is not None:
            return chosen_seller

    raise ValueError("Unable to find a seller.")


def handle_transaction(
    industry: int,
    buyer: Agent,
    buyer_ind: int,
    seller: Agent,
    seller_ind: int,
) -> None:
    if seller.transactor_seller_states["Value Type"] != ValueType.REAL:
        raise ValueError("Nominal seller value type not supported.")

    # Price
    price = seller.transactor_seller_states["Prices"][seller_ind]

    # Determine the amount of the transaction
    if buyer.transactor_buyer_states["Value Type"] == ValueType.REAL:
        real_value = min(
            seller.transactor_seller_states["Remaining Goods"][seller_ind],
            buyer.transactor_buyer_states["Remaining Goods"][buyer_ind, industry],
        )
        buyer.transactor_buyer_states["Remaining Goods"][buyer_ind, industry] -= real_value
    else:
        real_value = min(
            seller.transactor_seller_states["Remaining Goods"][seller_ind],
            buyer.transactor_buyer_states["Remaining Goods"][buyer_ind, industry] / price,
        )
        buyer.transactor_buyer_states["Remaining Goods"][buyer_ind, industry] -= price * real_value

    # Check if we're close to 0 for the buyer
    if np.round(buyer.transactor_buyer_states["Remaining Goods"][buyer_ind, industry], 6) == 0.0:
        buyer.transactor_buyer_states["Remaining Goods"][buyer_ind, industry] = 0.0

    # Record the transaction for the seller
    seller.transactor_seller_states["Remaining Goods"][seller_ind] -= real_value
    if np.round(seller.transactor_seller_states["Remaining Goods"][seller_ind], 6) == 0.0:
        seller.transactor_seller_states["Remaining Goods"][seller_ind] = 0.0

    # Record the amount sold by the seller
    seller.transactor_seller_states["Real Amount sold"][seller_ind] += real_value
    if buyer.country_name == seller.country_name:
        seller.transactor_seller_states["Real Domestic Amount sold"][seller_ind] += real_value
    else:
        seller.transactor_seller_states["Real Foreign Amount sold"][seller_ind] += real_value

    # Record the amount spent by the buyer
    buyer.transactor_buyer_states["Nominal Amount spent"][buyer_ind, industry] += price * real_value
    buyer.transactor_buyer_states["Real Amount bought"][buyer_ind, industry] += real_value
    if buyer.country_name == seller.country_name:
        buyer.transactor_buyer_states["Nominal Amount spent on Domestic Goods"][buyer_ind, industry] += (
            price * real_value
        )
        buyer.transactor_buyer_states["Domestic Real Amount bought"][buyer_ind, industry] += real_value
    else:
        buyer.transactor_buyer_states["Nominal Amount spent on Foreign Goods"][buyer_ind, industry] += (
            price * real_value
        )
        buyer.transactor_buyer_states["Foreign Real Amount bought"][buyer_ind, industry] += real_value


def handle_hypothetical_transaction(
    industry: int,
    buyer: Agent,
    buyer_ind: int,
    seller: Agent,
    seller_ind: int,
) -> None:
    if seller.transactor_seller_states["Value Type"] != ValueType.REAL:
        raise ValueError("Nominal seller value type not supported.")

    # Price
    price = seller.transactor_seller_states["Prices"][seller_ind]

    # Process the transaction
    if buyer.transactor_buyer_states["Value Type"] == ValueType.REAL:
        seller.transactor_seller_states["Real Excess Demand"][seller_ind] += buyer.transactor_buyer_states[
            "Remaining Excess Goods"
        ][buyer_ind, industry]
    elif buyer.transactor_buyer_states["Value Type"] == ValueType.NOMINAL:
        seller.transactor_seller_states["Real Excess Demand"][seller_ind] += (
            buyer.transactor_buyer_states["Remaining Excess Goods"][buyer_ind, industry] / price
        )
    buyer.transactor_buyer_states["Remaining Excess Goods"][buyer_ind, industry] = 0.0


def update_supply_chain(
    current_supply_chain: dict[int, dict[Agent, dict[int, list[Tuple[Agent, int]]]]],
    industry: int,
    buyer: Agent,
    buyer_ind: int,
    seller: Agent,
    seller_ind: int,
) -> None:
    if buyer not in current_supply_chain[industry].keys():
        current_supply_chain[industry][buyer] = {}
    if buyer_ind in current_supply_chain[industry][buyer].keys():
        current_supply_chain[industry][buyer][buyer_ind] += [(seller, seller_ind)]
    else:
        current_supply_chain[industry][buyer][buyer_ind] = [(seller, seller_ind)]


def clean_rounding_errors(goods_market_participants: dict[str, list[Agent]], decimals: int = 6) -> None:
    for country_name in goods_market_participants.keys():
        for transactor in goods_market_participants[country_name]:
            transactor.transactor_seller_states["Remaining Goods"] = np.round(
                transactor.transactor_seller_states["Remaining Goods"], decimals
            )
            transactor.transactor_buyer_states["Remaining Goods"] = np.round(
                transactor.transactor_buyer_states["Remaining Goods"], decimals
            )
