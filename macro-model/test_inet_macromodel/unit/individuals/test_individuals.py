from configurations import IndividualsConfiguration
from individuals import Individuals


class TestIndividuals:
    def test__init(self, datawrapper):
        synthetic_population = datawrapper.synthetic_countries["FRA"].population

        test_individuals = Individuals.from_pickled_agent(
            synthetic_population=synthetic_population,
            configuration=IndividualsConfiguration(),
            country_name="FRA",
            all_country_names=["FRA"],
            n_industries=18,
            scale=10_000,
        )

        states = [
            "Gender",
            "Age",
            "Education",
            "Activity Status",
            "Employment Industry",
            "Income",
            "Employee Income",
            "Income from Unemployment Benefits",
            "Corresponding Household ID",
            "Corresponding Firm ID",
        ]

        assert set(states).issubset(set(test_individuals.states.keys()))

        ts_keys = [
            "n_individuals",
            "employee_income",
            "income_from_unemployment_benefits",
            "income",
            "labour_inputs",
            "reservation_wages",
        ]

        assert set(ts_keys).issubset(set(test_individuals.ts.get_keys()))

    #
    # def test__individuals_states(self, test_individuals):
    #     assert test_individuals is not None
    #     for state in [
    #         "Gender",
    #         "Age",
    #         "Education",
    #         "Activity Status",
    #         "Employment Industry",
    #         "Income",
    #         "Employee Income",
    #         "Income from Unemployment Benefits",
    #         "Corresponding Household ID",
    #         "Corresponding Firm ID",
    #     ]:
    #         assert state in test_individuals.states.keys()
    #
    # def test__individuals_ts(self, test_individuals):
    #     for ts_key in [
    #         "n_individuals",
    #         "employee_income",
    #         "income_from_unemployment_benefits",
    #         "income",
    #         "labour_inputs",
    #         "reservation_wages",
    #     ]:
    #         assert ts_key in test_individuals.ts.get_keys()
