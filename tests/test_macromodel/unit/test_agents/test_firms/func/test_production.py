import numpy as np

from macromodel.agents.firms.func.production import BundledLeontief, PureLeontief
from macromodel.agents.firms.func.target_intermediate_inputs import (
    BundleWeightedTargetIntermediateInputsSetter,
)
from macromodel.agents.firms.utils.create_bundle_matrix import create_bundle_matrix
from macromodel.configurations.firms_configuration import create_good_bundle


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

    def test__compute_production_with_tfp_none(self):
        """Test backward compatibility when TFP is not provided."""
        assert np.allclose(
            PureLeontief().compute_production(
                desired_production=np.array([10.0, 10.0]),
                current_labour_inputs=np.array([9.0, 11.0]),
                current_limiting_intermediate_inputs=np.array([9.0, 11.0]),
                current_limiting_capital_inputs=np.array([6.0, 8.0]),
                tfp_multiplier=None,  # Explicitly None
            ),
            np.array([6.0, 8.0]),
        )

    def test__compute_production_with_tfp_unity(self):
        """Test that TFP=1.0 gives same result as no TFP."""
        tfp_unity = np.array([1.0, 1.0])
        assert np.allclose(
            PureLeontief().compute_production(
                desired_production=np.array([10.0, 10.0]),
                current_labour_inputs=np.array([9.0, 11.0]),
                current_limiting_intermediate_inputs=np.array([9.0, 11.0]),
                current_limiting_capital_inputs=np.array([6.0, 8.0]),
                tfp_multiplier=tfp_unity,
            ),
            np.array([6.0, 8.0]),
        )

    def test__compute_production_with_tfp_boost(self):
        """Test that TFP>1.0 scales labour in compute_production().

        Note: Limiting stock (intermediate + capital inputs) is pre-scaled by TFP
        in set_targets(), so compute_production() only scales labour here.
        This test passes pre-scaled limiting inputs to reflect real usage.
        """
        tfp_boost = np.array([1.5, 1.2])  # 50% and 20% productivity boost
        # Simulate pre-scaled limiting inputs (as done in set_targets())
        # Raw limiting intermediate = [9, 11], raw limiting capital = [6, 8]
        pre_scaled_limiting_intermediate = np.array([9.0, 11.0]) * tfp_boost  # [13.5, 13.2]
        pre_scaled_limiting_capital = np.array([6.0, 8.0]) * tfp_boost  # [9.0, 9.6]
        result = PureLeontief().compute_production(
            desired_production=np.array([10.0, 10.0]),
            current_labour_inputs=np.array([9.0, 11.0]),
            current_limiting_intermediate_inputs=pre_scaled_limiting_intermediate,
            current_limiting_capital_inputs=pre_scaled_limiting_capital,
            tfp_multiplier=tfp_boost,
        )
        # Labour is scaled in compute_production: 9*1.5=13.5, 11*1.2=13.2
        # Limiting stock is already pre-scaled: min(intermediate, capital) = [9.0, 9.6]
        # Final: min([10, 13.5, 9.0], [10, 13.2, 9.6]) = [9.0, 9.6]
        assert np.allclose(result, np.array([9.0, 9.6]))

    def test__compute_production_tfp_respects_target(self):
        """Test that TFP doesn't allow production above target."""
        tfp_high = np.array([2.0, 2.0])  # Double productivity
        # Simulate pre-scaled limiting inputs (as done in set_targets())
        pre_scaled_limiting_intermediate = np.array([9.0, 11.0]) * tfp_high  # [18.0, 22.0]
        pre_scaled_limiting_capital = np.array([6.0, 8.0]) * tfp_high  # [12.0, 16.0]
        result = PureLeontief().compute_production(
            desired_production=np.array([5.0, 7.0]),  # Low targets
            current_labour_inputs=np.array([9.0, 11.0]),
            current_limiting_intermediate_inputs=pre_scaled_limiting_intermediate,
            current_limiting_capital_inputs=pre_scaled_limiting_capital,
            tfp_multiplier=tfp_high,
        )
        # Despite high TFP, production is limited by desired_production
        assert np.allclose(result, np.array([5.0, 7.0]))

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
        default_bundle = create_good_bundle(5)
        current_target_production = np.full(n_industries, 1)
        intermediate_inputs_productivity_matrix = np.full((n_industries, n_industries), 1)
        prev_intermediate_inputs_stock = np.full((n_industries, n_industries), 1)
        initial_intermediate_inputs_stock = np.full((n_industries, n_industries), 1)
        prev_production = np.full(n_industries, 1)
        initial_production = np.full(n_industries, 1)
        previous_good_prices = np.full(n_industries, 1)
        substitution_bundle_matrix = create_bundle_matrix(np.array(default_bundle))
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


class TestBundledLeontiefUsage:
    """Test that BundledLeontief distributes input usage within bundles
    proportionally to stock, not at fixed Leontief rates."""

    def test_usage_distributed_by_stock_within_bundle(self):
        """If two goods are in a bundle and one has 3x the stock,
        it should bear 3x the usage. Before the fix, both goods
        were used at the same fixed Leontief rate regardless of stock."""
        n_firms = 2
        n_goods = 3  # goods 0 and 1 in a bundle, good 2 is singleton

        # Leontief coefficients: each firm needs 1 unit of each good per unit of output
        productivity = np.full((n_firms, n_goods), 1.0)

        # Firm 0 has substituted toward good 0 (3x stock vs good 1)
        # Firm 1 has equal stock
        stock = np.array([
            [3.0, 1.0, 2.0],  # firm 0: 3x more of good 0 than good 1
            [2.0, 2.0, 2.0],  # firm 1: equal stock
        ])

        production = np.array([2.0, 2.0])

        # Bundle matrix: goods 0,1 in bundle 0; good 2 in bundle 1
        # Shape (n_goods, n_bundles) = (3, 2)
        bundle_matrix = np.array([
            [0.5, 0.0],  # good 0 in bundle 0
            [0.5, 0.0],  # good 1 in bundle 0
            [0.0, 1.0],  # good 2 in bundle 1 (singleton)
        ])

        criticality = np.ones((n_firms, n_goods))

        bundled = BundledLeontief()
        used = bundled.compute_intermediate_inputs_used(
            realised_production=production,
            intermediate_inputs_productivity_matrix=productivity,
            intermediate_inputs_stock=stock,
            goods_criticality_matrix=criticality,
            substitution_bundle_matrix=bundle_matrix,
        )

        # Total bundle usage for bundle 0: production/coeff[0] + production/coeff[1] = 2+2 = 4
        # Firm 0: stock shares are 3/4 and 1/4 → used = [3.0, 1.0]
        # Firm 1: stock shares are 2/4 and 2/4 → used = [2.0, 2.0]
        # Good 2 (singleton): unchanged at production/coeff = 2.0
        expected = np.array([
            [3.0, 1.0, 2.0],
            [2.0, 2.0, 2.0],
        ])
        assert np.allclose(used, expected)

    def test_usage_total_preserved_within_bundle(self):
        """Total usage across bundle members should equal the sum of
        individual Leontief requirements, just redistributed."""
        n_firms = 3
        n_goods = 4  # goods 0,1,2 in a bundle, good 3 singleton

        productivity = np.array([
            [2.0, 4.0, 5.0, 3.0],
            [2.0, 4.0, 5.0, 3.0],
            [2.0, 4.0, 5.0, 3.0],
        ])

        stock = np.array([
            [10.0, 5.0, 5.0, 8.0],
            [1.0, 1.0, 18.0, 8.0],
            [0.0, 0.0, 0.0, 8.0],  # no stock at all
        ])

        production = np.array([10.0, 10.0, 10.0])

        bundle_matrix = np.array([
            [1 / 3, 0.0],
            [1 / 3, 0.0],
            [1 / 3, 0.0],
            [0.0, 1.0],
        ])

        criticality = np.ones((n_firms, n_goods))

        bundled = BundledLeontief()
        used = bundled.compute_intermediate_inputs_used(
            realised_production=production,
            intermediate_inputs_productivity_matrix=productivity,
            intermediate_inputs_stock=stock,
            goods_criticality_matrix=criticality,
            substitution_bundle_matrix=bundle_matrix,
        )

        # Total Leontief usage per firm for bundle goods: 10/2 + 10/4 + 10/5 = 5+2.5+2 = 9.5
        bundle_total = used[:, :3].sum(axis=1)
        assert np.allclose(bundle_total, 9.5)

        # Singleton good 3 unchanged: 10/3
        assert np.allclose(used[:, 3], 10.0 / 3.0)

        # Firm 2 has zero stock → equal distribution: 9.5/3 each
        assert np.allclose(used[2, :3], 9.5 / 3.0)
