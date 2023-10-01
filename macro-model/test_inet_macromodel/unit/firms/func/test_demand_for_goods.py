import numpy as np

from inet_macromodel.firms.func.demand_for_goods import (
    DefaultDemandSetter,
    DemandExcessSetter,
)


class TestDemandSetter:
    def test__compute_demand(self):
        assert np.allclose(
            DefaultDemandSetter().compute_demand(
                sell_real=np.array([1.0, 2.0]),
                excess_demand=np.array([0.0, 0.5]),
            ),
            np.array([1.0, 2.0]),
        )
        assert np.allclose(
            DemandExcessSetter().compute_demand(
                sell_real=np.array([1.0, 2.0]),
                excess_demand=np.array([0.0, 0.5]),
            ),
            np.array([1.0, 2.5]),
        )
