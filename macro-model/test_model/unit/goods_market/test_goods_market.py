import numpy as np

from model.agents.agent import Agent
from model.timeseries import TimeSeries
from model.goods_market.value_type import ValueType

from typing import Tuple


def create_test_transactor(
    buy_value_type: ValueType,
    sell_value_type: ValueType,
    buy_priority: int,
    sell_priority: int,
    n_transactors_buy: int,
    n_transactors_sell: int,
) -> Agent:
    return Agent(
        country_name="GER",
        all_country_names=["GER"],
        year=2014,
        t_max=12,
        n_industries=18,
        n_transactors_buy=n_transactors_buy,
        n_transactors_sell=n_transactors_sell,
        functions={},
        parameters={},
        ts=TimeSeries(),
        states={},
        transactor_settings={
            "Buyer Value Type": buy_value_type,
            "Seller Value Type": sell_value_type,
            "Buyer Priority": buy_priority,
            "Seller Priority": sell_priority,
        },
    )


def create_test_transactors() -> Tuple[Agent, Agent]:
    # A few firms
    firms = create_test_transactor(
        buy_value_type=ValueType.REAL,
        sell_value_type=ValueType.REAL,
        buy_priority=1,
        sell_priority=1,
        n_transactors_buy=3,
        n_transactors_sell=3,
    )
    firms.set_goods_to_buy(np.array([[5.0, 2.0], [5.0, 3.0], [0.0, 0.0]]))
    firms.set_goods_to_sell(np.array([35.0, 5.0, 10.0]))
    firms.set_prices(np.array([1.0, 1.0, 2.0]))
    firms.set_seller_industries(np.array([0, 0, 1]))
    firms.set_exchange_rate(1.0)

    # A few households
    households = create_test_transactor(
        buy_value_type=ValueType.NOMINAL,
        sell_value_type=ValueType.NONE,
        buy_priority=0,
        sell_priority=0,
        n_transactors_buy=3,
        n_transactors_sell=3,
    )
    households.set_goods_to_buy(np.array([[5.0, 5.0], [3.0, 2.0], [2.0, 3.0]]))
    households.set_goods_to_sell(np.array([0.0, 0.0, 0.0]))
    households.set_prices(np.array([0.0, 0.0, 0.0]))
    households.set_exchange_rate(1.0)

    return firms, households


def check_things_adding_up(firms: Agent, households: Agent) -> None:
    nominal_sold = firms.ts.current("nominal_amount_sold_in_lcu").sum()
    nominal_bought = (
        firms.ts.current("nominal_amount_spent_in_lcu").sum()
        + households.ts.current("nominal_amount_spent_in_lcu").sum()
    )
    assert np.isclose(nominal_sold, nominal_bought, atol=1e-1)


def check_excess_demand(firms: Agent, households: Agent) -> None:
    prices = np.array([1.0, 2.0])
    for g in [0, 1]:
        nominal_excess_demand = (
            prices[g] * firms.ts.current("real_excess_demand")[firms.transactor_seller_states["Industries"] == g].sum()
        )
        nominal_supply = prices[g] * (
            firms.transactor_seller_states["Initial Goods"][firms.transactor_seller_states["Industries"] == g].sum()
        )
        nominal_demand = (
            prices[g] * firms.transactor_buyer_states["Initial Goods"][:, g].sum()
            + households.transactor_buyer_states["Initial Goods"][:, g].sum()
        )
        if nominal_demand > nominal_supply:
            assert np.allclose(nominal_excess_demand, nominal_demand - nominal_supply, atol=1e-1)
        else:
            assert np.allclose(nominal_excess_demand, 0.0, atol=1e-1)


class TestGoodsMarket:
    def test__transaction(self, test_goods_market):
        # Add agents
        firms, households = create_test_transactors()
        test_goods_market.functions["clearing"].initiate_agents(
            n_industries=2,
            goods_market_participants={
                "GER": [firms, households],
            },
        )

        # Clear
        test_goods_market.prepare()
        test_goods_market.clear()
        test_goods_market.record()

        # Check basics
        check_things_adding_up(firms, households)
        check_excess_demand(firms, households)
