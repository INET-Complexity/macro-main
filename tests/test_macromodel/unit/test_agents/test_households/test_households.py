import numpy as np

from macro_data.readers.emission_fraction.emission_fraction_reader import EmissionFractions


def _setup_emission_ts(households, n_hh, n_industries):
    """Inject a non-NaN spending matrix and emission timeseries keys."""
    households.ts.dicts["nominal_amount_spent_in_lcu"][-1] = np.full((n_hh, n_industries), 10.0)
    for key in [
        "consumption_emissions",
        "investment_emissions",
        "consumption_emissions_by_good",
        "investment_emissions_by_good",
        "coal_consumption_emissions",
        "oil_consumption_emissions",
        "gas_consumption_emissions",
        "refined_products_consumption_emissions",
        "coal_investment_emissions",
        "oil_investment_emissions",
        "gas_investment_emissions",
        "refined_products_investment_emissions",
    ]:
        size = n_industries if "by_good" in key else n_hh
        households.ts.dicts[key] = [np.zeros(size)]


class TestHouseholds:
    def test__create(self, test_households):
        assert test_households.country_name == "FRA"

    def test__households_states(self, test_households):
        assert test_households is not None
        for state in [
            "saving_rates_model",
            "social_transfers_model",
            "average_saving_rate",
            "Type",
            "Corresponding Bank ID",
            "Corresponding Inhabited House ID",
            "Corresponding Property Owner",
            "Tenure Status of the Main Residence",
            "corr_individuals",
            "corr_renters",
        ]:
            assert state in test_households.states.keys()

    # def test__households_ts(self, test_households):
    #     for ts_key in [
    #         "n_households",
    #         "target_consumption_before_ce",
    #         "target_consumption_ce",
    #         "target_consumption",
    #         "amount_bought",
    #         "consumption",
    #         "investment_in_other_real_assets",
    #         "income",
    #         "income_employee",
    #         "income_social_transfers",
    #         "income_rental",
    #         "price_paid_for_property",
    #         "rent",
    #         "max_price_willing_to_pay",
    #         "max_rent_willing_to_pay",
    #         "wealth",
    #         "wealth_real_assets",
    #         "wealth_main_residence",
    #         "wealth_other_properties",
    #         "wealth_other_real_assets",
    #         "wealth_deposits",
    #         "wealth_other_financial_assets",
    #         "wealth_financial_assets",
    #         "payday_loan_debt",
    #         "consumption_expansion_loan_debt",
    #         "mortgage_debt",
    #         "debt",
    #         "net_wealth",
    #         "target_payday_loans",
    #         "received_payday_loans",
    #         "target_consumption_expansion_loans",
    #         "received_consumption_expansion_loans",
    #         "target_mortgage",
    #         "received_mortgages",
    #         "debt_installments",
    #         "interest_paid_on_deposits",
    #         "interest_paid_on_loans",
    #         "interest_paid",
    #     ]:
    #         assert ts_key in test_households.ts.get_keys()
    #
    # def test__get_saving_rates_by_household(self, test_households):
    #     assert np.allclose(test_households.get_saving_rates_by_household(), np.full(18, 0.2))
    #
    # def test__get_social_transfers_by_household(self, test_households):
    #     assert np.allclose(
    #         test_households.compute_social_transfer_income(
    #             total_other_social_transfers=1000.0,
    #             central_government_init={
    #                 "functions": {
    #                     "household_social_transfers": {
    #                         "parameters": {
    #                             "independents": {"value": []},
    #                             "steps": {"value": 1},
    #                         }
    #                     }
    #                 }
    #             },
    #         ),
    #         np.full(18, 55.55555556),
    #     )


class TestHouseholdsUpdateConsumptionEmissions:
    """Tests for the emission multiplier logic in update_consumption_and_investment."""

    def test__appends_consumption_emissions(self, test_households):
        n_hh = len(test_households.states["Type"])
        n_industries = test_households.n_industries
        _setup_emission_ts(test_households, n_hh, n_industries)
        n_before = len(test_households.ts.consumption_emissions)
        emitting_indices = np.array([0, 1, 2, 3])

        test_households.update_consumption_and_investment(
            tau_vat=0.0,
            tau_cf=0.0,
            add_emissions=True,
            readjusted_factors=np.ones(4),
            emitting_indices=emitting_indices,
        )

        assert len(test_households.ts.consumption_emissions) == n_before + 1
        assert len(test_households.ts.investment_emissions) == n_before + 1

    def test__consumption_emissions_by_good_appended(self, test_households):
        n_hh = len(test_households.states["Type"])
        n_industries = test_households.n_industries
        _setup_emission_ts(test_households, n_hh, n_industries)
        emitting_indices = np.array([0, 1, 2, 3])

        test_households.update_consumption_and_investment(
            tau_vat=0.0,
            tau_cf=0.0,
            add_emissions=True,
            readjusted_factors=np.ones(4),
            emitting_indices=emitting_indices,
        )

        assert len(test_households.ts.consumption_emissions_by_good) == 2
        assert len(test_households.ts.investment_emissions_by_good) == 2

    def test__all_ones_multiplier_matches_no_multiplier(self, test_households):
        """Consumption fractions of 1.0 everywhere should give same emissions as no multiplier."""
        n_hh = len(test_households.states["Type"])
        n_industries = test_households.n_industries
        _setup_emission_ts(test_households, n_hh, n_industries)
        emitting_indices = np.array([0, 1, 2, 3])
        readjusted_factors = np.ones(4)

        test_households.update_consumption_and_investment(
            tau_vat=0.0,
            tau_cf=0.0,
            add_emissions=True,
            readjusted_factors=readjusted_factors,
            emitting_indices=emitting_indices,
            use_emission_multiplier=False,
        )
        baseline = test_households.ts.consumption_emissions[-1].copy()

        consumption_ones = np.ones((1, n_industries))
        investment_ones = np.ones((1, n_industries))
        test_households.emission_fractions = EmissionFractions(consumption=consumption_ones, investment=investment_ones)
        test_households.update_consumption_and_investment(
            tau_vat=0.0,
            tau_cf=0.0,
            add_emissions=True,
            readjusted_factors=readjusted_factors,
            emitting_indices=emitting_indices,
            use_emission_multiplier=True,
        )

        assert np.allclose(baseline, test_households.ts.consumption_emissions[-1])

    def test__zero_multiplier_yields_zero_emissions(self, test_households):
        """Consumption fractions of 0.0 should zero out emissions."""
        n_hh = len(test_households.states["Type"])
        n_industries = test_households.n_industries
        _setup_emission_ts(test_households, n_hh, n_industries)
        emitting_indices = np.array([0, 1, 2, 3])

        consumption_zeros = np.zeros((1, n_industries))
        investment_zeros = np.zeros((1, n_industries))
        test_households.emission_fractions = EmissionFractions(
            consumption=consumption_zeros, investment=investment_zeros
        )
        test_households.update_consumption_and_investment(
            tau_vat=0.0,
            tau_cf=0.0,
            add_emissions=True,
            readjusted_factors=np.ones(4),
            emitting_indices=emitting_indices,
            use_emission_multiplier=True,
        )

        assert np.allclose(test_households.ts.consumption_emissions[-1], 0.0)
        assert np.allclose(test_households.ts.investment_emissions[-1], 0.0)

    def test__ch4_by_good_populated_when_ch4_factors_provided(self, test_households):
        n_hh = len(test_households.states["Type"])
        n_industries = test_households.n_industries
        _setup_emission_ts(test_households, n_hh, n_industries)
        test_households.ts.dicts["consumption_emissions_ch4_by_good"] = [np.zeros(n_industries)]
        test_households.ts.dicts["investment_emissions_ch4_by_good"] = [np.zeros(n_industries)]
        emitting_indices = np.array([0, 1, 2, 3])

        test_households.update_consumption_and_investment(
            tau_vat=0.0,
            tau_cf=0.0,
            add_emissions=True,
            readjusted_factors=np.ones(4),
            emitting_indices=emitting_indices,
            readjusted_factors_ch4=np.ones(4),
            emitting_indices_ch4=np.array([0, 1, 2, 3]),
        )

        assert len(test_households.ts.consumption_emissions_ch4_by_good) == 2
        assert len(test_households.ts.investment_emissions_ch4_by_good) == 2
