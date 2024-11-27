import numpy as np

from macromodel.banks.func.demography import NoBankDemography


class TestBankDemography:
    def test__handle_bank_insolvency(self):
        NoBankDemography(solvency_ratio=0.05).handle_bank_insolvency(
            current_bank_equity=np.array([100.0, 100.0]),
            current_bank_deposits=np.array([50.0, 50.0]),
            current_bank_loans=np.array([10.0, 20.0]),
            is_insolvent=np.array([False, False]),
        )
