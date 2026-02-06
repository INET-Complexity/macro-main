import numpy as np

from macromodel.agents.central_government.func.social_benefits import (
    DefaultSocialBenefitsSetter,
)


class TestSocialBenefitsSetter:
    def test__compute_unemployment_benefits(self):
        assert (
            DefaultSocialBenefitsSetter().compute_unemployment_benefits(
                prev_unemployment_benefits=100.0,
                historic_ppi_inflation=np.array([0.01, 0.02]),
                current_unemployment_rate=0.1,
                current_estimated_growth=0.0,
                model=None,
            )
            == 100.0
        )

    def test__compute_regular_transfer_to_households(self):
        assert (
            DefaultSocialBenefitsSetter().compute_regular_transfer_to_households(
                prev_regular_transfer_to_households=100.0,
                historic_ppi_inflation=np.array([0.01, 0.02]),
                current_unemployment_rate=0.1,
                current_estimated_growth=0.0,
                model=None,
            )
            == 100.0
        )
