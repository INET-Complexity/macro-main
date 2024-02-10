from macromodel.government_entities.func.consumption import (
    DefaultGovernmentConsumptionSetter,
)


class TestGovernmentConsumptionSetter:
    def test__compute_desired_consumption(self):
        assert (
            DefaultGovernmentConsumptionSetter().compute_target_consumption(
                previous_desired_government_consumption=100.0,
                model=None,
            )
            == 100.0
        )
