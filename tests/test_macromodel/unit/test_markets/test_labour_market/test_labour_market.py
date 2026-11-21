from types import SimpleNamespace

import numpy as np

from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.markets.labour_market.func.clearing import PolednaLabourMarketClearer


class _DummyTS:
    def __init__(self, values):
        self._values = values

    def current(self, name):
        return self._values[name]


class TestLabourMarket:
    def test_poledna_clear_marks_unassigned_active_individuals_as_unemployed(self):
        clearer = PolednaLabourMarketClearer(
            hiring_speed=0.0,
            firing_speed=0.0,
            random_firing_probability=0.0,
            sorted_firing=False,
            optimised_hiring=True,
            allow_switching_industries=True,
            consider_reservation_wages=False,
            firing_cost_fraction=0.0,
            hiring_cost_fraction=0.0,
            individuals_quitting=False,
            individuals_quitting_temperature=0.0,
            compare_with_normalised_inputs=False,
            round_target_employment=False,
        )

        firms = SimpleNamespace(
            ts=_DummyTS(
                {
                    "labour_inputs": np.array([1.0]),
                    "desired_labour_inputs": np.array([1.0]),
                    "n_firms": 1,
                }
            ),
            states={
                "Employments": [[0]],
                "Industry": np.array([0]),
                "Labour Productivity by Industry": np.array([1.0]),
            },
        )
        households = SimpleNamespace(ts=_DummyTS({"wealth": np.array([0.0, 0.0])}))
        individuals = SimpleNamespace(
            ts=_DummyTS(
                {
                    "labour_inputs": np.array([1.0, 0.0]),
                    "employee_income": np.array([1.0, 0.0]),
                    "reservation_wages": np.array([0.0, 0.0]),
                }
            ),
            states={
                "Activity Status": np.array([ActivityStatus.EMPLOYED, ActivityStatus.EMPLOYED], dtype=object),
                "Employment Industry": np.array([0, 0]),
                "Corresponding Firm ID": np.array([0, -1]),
                "Corresponding Household ID": np.array([0, 1]),
                "Offered Wage of Accepted Job": np.zeros(2),
            },
        )

        clearer.clear(firms=firms, households=households, individuals=individuals)

        assert individuals.states["Activity Status"][0] == ActivityStatus.EMPLOYED
        assert individuals.states["Activity Status"][1] == ActivityStatus.UNEMPLOYED
