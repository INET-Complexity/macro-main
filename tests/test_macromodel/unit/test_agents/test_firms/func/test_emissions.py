import numpy as np

from macro_data.readers.emission_fraction.emission_fraction_reader import EmissionFractions


def _init_emission_ts(firms):
    """Inject emission timeseries keys so update_emissions can append to them."""
    n_firms = len(firms.states["Industry"])
    for key in [
        "inputs_emissions",
        "capital_emissions",
        "inputs_emissions_ch4",
        "capital_emissions_ch4",
        "coal_inputs_emissions",
        "gas_inputs_emissions",
        "oil_inputs_emissions",
        "refined_products_inputs_emissions",
        "coal_capital_emissions",
        "gas_capital_emissions",
        "oil_capital_emissions",
        "refined_products_capital_emissions",
    ]:
        firms.ts.dicts[key] = [np.zeros(n_firms)]


class TestFirmsUpdateEmissions:
    def test__appends_to_timeseries(self, test_firms):
        _init_emission_ts(test_firms)
        n_before = len(test_firms.ts.inputs_emissions)
        emitting_indices = np.array([0, 1, 2, 3])
        readjusted_factors = np.ones(4)

        test_firms.update_emissions(readjusted_factors, emitting_indices)

        assert len(test_firms.ts.inputs_emissions) == n_before + 1
        assert len(test_firms.ts.capital_emissions) == n_before + 1

    def test__disaggregated_emissions_appended(self, test_firms):
        _init_emission_ts(test_firms)
        emitting_indices = np.array([0, 1, 2, 3])

        test_firms.update_emissions(np.ones(4), emitting_indices)

        assert len(test_firms.ts.coal_inputs_emissions) == 2
        assert len(test_firms.ts.gas_inputs_emissions) == 2
        assert len(test_firms.ts.oil_inputs_emissions) == 2
        assert len(test_firms.ts.refined_products_inputs_emissions) == 2

    def test__all_ones_multiplier_matches_no_multiplier(self, test_firms):
        """CO2 fractions of 1.0 everywhere should produce the same result as no multiplier."""
        _init_emission_ts(test_firms)
        emitting_indices = np.array([0, 1, 2, 3])
        readjusted_factors = np.ones(4)
        n_industries = test_firms.n_industries

        test_firms.update_emissions(readjusted_factors, emitting_indices, use_emission_multiplier=False)
        baseline_inputs = test_firms.ts.inputs_emissions[-1].copy()
        baseline_capital = test_firms.ts.capital_emissions[-1].copy()

        co2_ones = np.ones((4, n_industries))
        test_firms.emission_fractions = EmissionFractions(co2=co2_ones)
        test_firms.update_emissions(readjusted_factors, emitting_indices, use_emission_multiplier=True)

        assert np.allclose(baseline_inputs, test_firms.ts.inputs_emissions[-1])
        assert np.allclose(baseline_capital, test_firms.ts.capital_emissions[-1])

    def test__zero_multiplier_yields_zero_emissions(self, test_firms):
        """CO2 fractions of 0.0 everywhere should zero out all emissions."""
        _init_emission_ts(test_firms)
        emitting_indices = np.array([0, 1, 2, 3])
        n_industries = test_firms.n_industries

        co2_zeros = np.zeros((4, n_industries))
        test_firms.emission_fractions = EmissionFractions(co2=co2_zeros)
        test_firms.update_emissions(np.ones(4), emitting_indices, use_emission_multiplier=True)

        assert np.allclose(test_firms.ts.inputs_emissions[-1], 0.0)
        assert np.allclose(test_firms.ts.capital_emissions[-1], 0.0)

    def test__ch4_timeseries_populated_when_factors_provided(self, test_firms):
        _init_emission_ts(test_firms)
        emitting_indices = np.array([0, 1, 2, 3])
        n_before = len(test_firms.ts.inputs_emissions_ch4)

        test_firms.update_emissions(
            readjusted_factors=np.ones(4),
            emitting_indices=emitting_indices,
            readjusted_factors_ch4=np.ones(4),
            emitting_indices_ch4=np.array([0, 1, 2, 3]),
        )

        assert len(test_firms.ts.inputs_emissions_ch4) == n_before + 1
        assert len(test_firms.ts.capital_emissions_ch4) == n_before + 1

    def test__no_multiplier_when_emission_fractions_none(self, test_firms):
        """With use_emission_multiplier=True but emission_fractions=None, falls back to no-multiplier path."""
        _init_emission_ts(test_firms)
        emitting_indices = np.array([0, 1, 2, 3])
        readjusted_factors = np.ones(4)

        test_firms.emission_fractions = None
        test_firms.update_emissions(readjusted_factors, emitting_indices, use_emission_multiplier=False)
        baseline = test_firms.ts.inputs_emissions[-1].copy()

        test_firms.update_emissions(readjusted_factors, emitting_indices, use_emission_multiplier=True)
        with_flag_no_fractions = test_firms.ts.inputs_emissions[-1].copy()

        assert np.allclose(baseline, with_flag_no_fractions)
