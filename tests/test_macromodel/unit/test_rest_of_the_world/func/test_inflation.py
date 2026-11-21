from macromodel.rest_of_the_world.func.inflation import DefaultRoWInflationSetter


class TestRoWInflationSetter:
    def test__compute_growth(self):
        assert DefaultRoWInflationSetter().compute_inflation(average_country_ppi_inflation=0.02) == 0.02
