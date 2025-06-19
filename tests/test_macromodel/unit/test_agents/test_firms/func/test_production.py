import numpy as np

from macromodel.agents.firms.func.production import PureLeontief, BundledLeontief
from macromodel.agents.firms.func.target_intermediate_inputs import BundleWeightedTargetIntermediateInputsSetter


class TestProductionSetter:
    def test__compute_production(self):
        assert np.allclose(
            PureLeontief().compute_production(
                desired_production=np.array([10.0, 10.0]),
                current_labour_inputs=np.array([9.0, 11.0]),
                current_limiting_intermediate_inputs=np.array([9.0, 11.0]),
                current_limiting_capital_inputs=np.array([6.0, 8.0]),
            ),
            np.array([6.0, 8.0]),
        )

    def test__compute_limiting_intermediate_inputs_stock(self):
        intermediate_inputs_productivity_matrix = np.array(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [np.inf, np.inf, np.inf, np.inf], [1.0, 1.0, 1.0, 1.0]]
        )
        intermediate_inputs_stock = np.array(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
        )
        intermediate_inputs_utilisation_rate = np.ones(4)
        goods_criticality_matrix = np.ones((4, 4))
        substitution_bundle_matrix = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        bundle_test = BundledLeontief()
        bundle_productivity = bundle_test.compute_limiting_intermediate_inputs_stock(
            intermediate_inputs_productivity_matrix=intermediate_inputs_productivity_matrix,
            intermediate_inputs_stock=intermediate_inputs_stock,
            intermediate_inputs_utilisation_rate=intermediate_inputs_utilisation_rate,
            goods_criticality_matrix=goods_criticality_matrix,
            substitution_bundle_matrix=substitution_bundle_matrix,
        )

        assert not np.isnan(bundle_productivity).any()
        assert not np.isinf(bundle_productivity).any()

    def test_target_intermediate_inputs_bundle_empty(self):
        n_industries = 5
        bundles = np.array([])
        current_target_production = np.full(n_industries, 1)
        intermediate_inputs_productivity_matrix = np.full((n_industries, n_industries), 1)
        prev_intermediate_inputs_stock = np.full((n_industries, n_industries), 1)
        initial_intermediate_inputs_stock = np.full((n_industries, n_industries), 1)
        prev_production = np.full(n_industries, 1)
        initial_production = np.full(n_industries, 1)
        previous_good_prices = np.full(n_industries, 1)
        substitution_bundle_matrix = bundles
        extra_taxes = None

        setter = BundleWeightedTargetIntermediateInputsSetter(
            target_intermediate_inputs_fraction=0.8,
            credit_gap_fraction=0.5,
            beta=2.0,  # Higher sensitivity to price/productivity differences
        )

        result = setter.compute_unconstrained_target_intermediate_inputs(
            current_target_production,
            intermediate_inputs_productivity_matrix,
            prev_intermediate_inputs_stock,
            initial_intermediate_inputs_stock,
            prev_production,
            initial_production,
            previous_good_prices,
            substitution_bundle_matrix,
            extra_taxes,
        )

        expected = np.full((n_industries, n_industries), 1)
        assert np.array_equal(result, expected)
