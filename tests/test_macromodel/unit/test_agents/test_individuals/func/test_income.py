import numpy as np

from macromodel.agents.individuals.func.income import DefaultIncomeSetter
from macromodel.agents.individuals.individual_properties import ActivityStatus


class TestIncomeSetter:
    def test__compute_income(self):
        assert np.allclose(
            DefaultIncomeSetter()
            .compute_income(
                current_individual_activity_status=np.array(
                    [
                        ActivityStatus.EMPLOYED,
                        ActivityStatus.EMPLOYED,
                        ActivityStatus.UNEMPLOYED,
                    ]
                ),
                current_wage=np.array(
                    [
                        10.0,
                        20.0,
                        5.0,
                    ]
                ),
                individual_social_benefits=np.array(
                    [
                        2.0,
                        2.0,
                        2.0,
                    ]
                ),
                firm_profits=np.array([]),
                bank_profits=np.array([]),
                corr_invested_banks=np.array([np.nan, np.nan, np.nan]),
                corr_invested_firms=np.array([np.nan, np.nan, np.nan]),
                cpi=1.0,
                dividend_payout_ratio=0.1,
                income_taxes=0.3,
                tau_firm=0.4,
            )
            .astype(float),
            np.array([12.0, 22.0, 2.0]),
        )
