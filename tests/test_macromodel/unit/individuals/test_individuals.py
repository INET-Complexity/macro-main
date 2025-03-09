class TestIndividuals:
    def test__individuals_states(self, test_individuals):
        assert test_individuals is not None
        for state in [
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
        ]:
            assert state in test_individuals.states.keys()

    def test__individuals_ts(self, test_individuals):
        for ts_key in [
            "n_individuals",
            "employee_income",
            "income_from_unemployment_benefits",
            "income",
            "labour_inputs",
            "reservation_wages",
        ]:
            assert ts_key in test_individuals.ts.get_keys()
