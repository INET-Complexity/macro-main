import numpy as np

from macromodel.agents.firms.func.bought_goods_distributor import (
    BoughtGoodsDistributorEvenly,
    BoughtGoodsDistributorIIPrio,
)


class TestBoughtGoodsDistributor:
    def test__distribute_bought_goods(self):
        new_ii, new_cap = BoughtGoodsDistributorIIPrio().distribute_bought_goods(
            desired_intermediate_inputs=np.array([6.0, 5.0]),
            desired_investment=np.array([7.0, 3.0]),
            buy_real=np.array([10.0, 10.0]),
        )
        assert np.allclose(new_ii, np.array([6.0, 5.0]))
        assert np.allclose(new_cap, np.array([4.0, 5.0]))

        new_ii, new_cap = BoughtGoodsDistributorEvenly().distribute_bought_goods(
            desired_intermediate_inputs=np.array([6.0, 5.0]),
            desired_investment=np.array([7.0, 3.0]),
            buy_real=np.array([10.0, 10.0]),
        )
        assert np.allclose(new_ii, np.array([4.61538462, 6.25]))
        assert np.allclose(new_cap, np.array([5.38461538, 3.75]))
