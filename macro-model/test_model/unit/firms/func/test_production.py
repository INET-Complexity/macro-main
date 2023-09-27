import numpy as np

from model.firms.func.production import (
    PureLeontief,
    CriticalAndImportantLeontief,
    CriticalLeontief,
    Linear,
)


class TestProductionSetter:
    def test__compute_production(self):
        assert np.allclose(
            PureLeontief().compute_production(
                desired_production=np.array([10.0, 10.0]),
                current_labour_inputs=np.array([9.0, 11.0]),
                intermediate_inputs_productivity_matrix=np.array([[10.0, 20.0], [np.inf, 30.0]]),
                intermediate_inputs_stock=np.array([[0.7, 2.0], [10.0, 0.6]]),
                capital_inputs_productivity_matrix=np.array([[2.0, 3.0], [4.0, 8.0]]),
                capital_inputs_stock=np.array([[6.0, 2.0], [2.0, 10.0]]),
                goods_criticality_matrix=np.array([[1.0, 0.0], [0.0, 0.5]]),
                intermediate_inputs_utilisation_rate=1.0,
                capital_inputs_utilisation_rate=1.0,
            ),
            np.array([6.0, 8.0]),
        )
        assert np.allclose(
            CriticalAndImportantLeontief().compute_production(
                desired_production=np.array([10.0, 10.0]),
                current_labour_inputs=np.array([9.0, 11.0]),
                intermediate_inputs_productivity_matrix=np.array([[10.0, 20.0], [np.inf, 30.0]]),
                intermediate_inputs_stock=np.array([[0.7, 2.0], [10.0, 0.6]]),
                capital_inputs_productivity_matrix=np.array([[2.0, 3.0], [4.0, 8.0]]),
                capital_inputs_stock=np.array([[6.0, 2.0], [2.0, 10.0]]),
                goods_criticality_matrix=np.array([[1.0, 0.0], [0.0, 0.5]]),
                intermediate_inputs_utilisation_rate=1.0,
                capital_inputs_utilisation_rate=1.0,
            ),
            np.array([7.0, 10.0]),
        )
        assert np.allclose(
            CriticalLeontief().compute_production(
                desired_production=np.array([10.0, 10.0]),
                current_labour_inputs=np.array([9.0, 11.0]),
                intermediate_inputs_productivity_matrix=np.array([[10.0, 20.0], [np.inf, 30.0]]),
                intermediate_inputs_stock=np.array([[0.7, 2.0], [10.0, 0.6]]),
                capital_inputs_productivity_matrix=np.array([[2.0, 3.0], [4.0, 8.0]]),
                capital_inputs_stock=np.array([[6.0, 2.0], [2.0, 10.0]]),
                goods_criticality_matrix=np.array([[1.0, 0.0], [0.0, 0.5]]),
                intermediate_inputs_utilisation_rate=1.0,
                capital_inputs_utilisation_rate=1.0,
            ),
            np.array([7.0, 10.0]),
        )
        assert np.allclose(
            Linear().compute_production(
                desired_production=np.array([10.0, 10.0]),
                current_labour_inputs=np.array([9.0, 11.0]),
                intermediate_inputs_productivity_matrix=np.array([[10.0, 20.0], [np.inf, 30.0]]),
                intermediate_inputs_stock=np.array([[0.7, 2.0], [10.0, 0.6]]),
                capital_inputs_productivity_matrix=np.array([[2.0, 3.0], [4.0, 8.0]]),
                capital_inputs_stock=np.array([[6.0, 2.0], [2.0, 10.0]]),
                goods_criticality_matrix=np.array([[1.0, 0.0], [0.0, 0.5]]),
                intermediate_inputs_utilisation_rate=1.0,
                capital_inputs_utilisation_rate=1.0,
            ),
            np.array([9.0, 10.0]),
        )
