import numpy as np
from pydantic import BaseModel, Field

from macromodel.agents.banks.func.interest_rates import (
    DefaultInterestRatesSetter,
    MarkUpInterestRatesSetter,
)
from macromodel.util.function_mapping import functions_from_model


class _InterestRateFunctionConfig(BaseModel):
    path_name: str = "interest_rates"
    name: str = "MarkUpInterestRatesSetter"
    parameters: dict = Field(
        default_factory=lambda: {
            "firm_short_spread": 0.03,
            "firm_long_spread": 0.04,
            "hh_consumption_spread": 0.05,
            "mortgage_spread": 0.02,
        }
    )


class _BankFunctionConfig(BaseModel):
    interest_rates: _InterestRateFunctionConfig = _InterestRateFunctionConfig()


class TestDefaultInterestRatesSetter:
    def test_keeps_existing_ect_formula(self):
        setter = DefaultInterestRatesSetter()

        prev_rates = np.array([0.03, 0.04])
        result = setter.get_interest_rates_on_short_term_firm_loans(
            central_bank_policy_rate=0.02,
            prev_interest_rates_on_short_term_firm_loans=prev_rates,
            firm_pt=0.8,
            firm_ect=0.5,
            firm_short_spread=np.array([0.10, 0.10]),
        )

        np.testing.assert_allclose(result, prev_rates + 0.5 * (prev_rates - 0.8 * 0.02))


class TestMarkUpInterestRatesSetter:
    def test_selects_from_model_with_constructor_overrides(self):
        functions = functions_from_model(
            model=_BankFunctionConfig(),
            loc="macromodel.agents.banks",
        )

        setter = functions["interest_rates"]
        assert isinstance(setter, MarkUpInterestRatesSetter)
        assert setter.firm_short_spread == 0.03
        assert setter.firm_long_spread == 0.04
        assert setter.hh_consumption_spread == 0.05
        assert setter.mortgage_spread == 0.02

    def test_constructor_overrides_bank_state_spreads(self):
        setter = MarkUpInterestRatesSetter(
            firm_short_spread=0.03,
            firm_long_spread=0.04,
            hh_consumption_spread=0.05,
            mortgage_spread=0.02,
        )

        np.testing.assert_allclose(
            setter.get_interest_rates_on_short_term_firm_loans(
                central_bank_policy_rate=0.01,
                prev_interest_rates_on_short_term_firm_loans=np.array([0.07, 0.08]),
                firm_pt=0.0,
                firm_ect=0.0,
                firm_short_spread=np.array([0.10, 0.11]),
            ),
            np.array([0.04, 0.04]),
        )
        np.testing.assert_allclose(
            setter.get_interest_rates_on_long_term_firm_loans(
                central_bank_policy_rate=0.01,
                prev_interest_rates_on_long_term_firm_loans=np.array([0.07, 0.08]),
                firm_pt=0.0,
                firm_ect=0.0,
                firm_long_spread=np.array([0.10, 0.11]),
            ),
            np.array([0.05, 0.05]),
        )
        np.testing.assert_allclose(
            setter.get_interest_rates_on_household_consumption_loans(
                central_bank_policy_rate=0.01,
                prev_interest_rate_on_hh_consumption_loans=np.array([0.07, 0.08]),
                hh_cons_pt=0.0,
                hh_cons_ect=0.0,
                hh_consumption_spread=np.array([0.10, 0.11]),
            ),
            np.array([0.06, 0.06]),
        )
        np.testing.assert_allclose(
            setter.get_interest_rate_on_mortgages(
                central_bank_policy_rate=0.01,
                prev_interest_rate_on_mortgages=np.array([0.07, 0.08]),
                hh_mortgage_pt=0.0,
                hh_mortgage_ect=0.0,
                mortgage_spread=np.array([0.10, 0.11]),
            ),
            np.array([0.03, 0.03]),
        )

    def test_falls_back_to_state_spreads_and_applies_floor(self):
        setter = MarkUpInterestRatesSetter()

        result = setter.get_interest_rates_on_short_term_firm_loans(
            central_bank_policy_rate=0.01,
            prev_interest_rates_on_short_term_firm_loans=np.array([0.07, 0.08]),
            firm_pt=0.0,
            firm_ect=0.0,
            firm_short_spread=np.array([-0.05, 0.02]),
        )

        np.testing.assert_allclose(result, np.array([0.00, 0.03]))

    def test_keeps_deposit_side_at_policy_rate(self):
        setter = MarkUpInterestRatesSetter()

        np.testing.assert_allclose(
            setter.compute_interest_rate_on_firm_deposits(
                central_bank_policy_rate=0.02,
                prev_interest_rate_on_firm_deposits=np.array([0.01, 0.03]),
                firm_pt=0.0,
                firm_ect=0.0,
            ),
            np.array([0.02, 0.02]),
        )
        np.testing.assert_allclose(
            setter.compute_overdraft_rate_on_household_deposits(
                central_bank_policy_rate=0.02,
                prev_overdraft_rate_on_hh_deposits=np.array([0.05, 0.06]),
                hh_cons_pt=0.0,
                hh_cons_ect=0.0,
            ),
            np.array([0.02, 0.02]),
        )


class TestBanksMarkUpInterestRates:
    def test_set_interest_rates_uses_bank_state_spreads(self, test_banks):
        test_banks.functions["interest_rates"] = MarkUpInterestRatesSetter()

        short_template = test_banks.ts.current("interest_rates_on_short_term_firm_loans")
        long_template = test_banks.ts.current("interest_rates_on_long_term_firm_loans")
        cons_template = test_banks.ts.current("interest_rates_on_household_consumption_loans")
        mortgage_template = test_banks.ts.current("interest_rates_on_mortgages")
        deposit_template = test_banks.ts.current("interest_rate_on_firm_deposits")

        test_banks.states["Firm Short Spread"] = np.full(short_template.shape, 0.01)
        test_banks.states["Firm Long Spread"] = np.full(long_template.shape, 0.015)
        test_banks.states["Household Consumption Spread"] = np.full(cons_template.shape, 0.02)
        test_banks.states["Mortgage Spread"] = np.full(mortgage_template.shape, 0.005)

        test_banks.set_interest_rates(central_bank_policy_rate=0.02)

        np.testing.assert_allclose(test_banks.ts.current("interest_rates_on_short_term_firm_loans"), 0.03)
        np.testing.assert_allclose(test_banks.ts.current("interest_rates_on_long_term_firm_loans"), 0.035)
        np.testing.assert_allclose(test_banks.ts.current("interest_rates_on_household_consumption_loans"), 0.04)
        np.testing.assert_allclose(test_banks.ts.current("interest_rates_on_mortgages"), 0.025)
        np.testing.assert_allclose(
            test_banks.ts.current("interest_rate_on_firm_deposits"), np.full(deposit_template.shape, 0.02)
        )
        np.testing.assert_allclose(
            test_banks.ts.current("interest_rate_on_household_deposits"),
            np.full(deposit_template.shape, 0.02),
        )
        np.testing.assert_allclose(
            test_banks.ts.current("overdraft_rate_on_firm_deposits"),
            np.full(deposit_template.shape, 0.02),
        )
        np.testing.assert_allclose(
            test_banks.ts.current("overdraft_rate_on_household_deposits"),
            np.full(deposit_template.shape, 0.02),
        )

    def test_set_interest_rates_prefers_constructor_overrides(self, test_banks):
        test_banks.functions["interest_rates"] = MarkUpInterestRatesSetter(
            firm_short_spread=0.03,
            firm_long_spread=0.04,
            hh_consumption_spread=0.05,
            mortgage_spread=0.02,
        )

        short_template = test_banks.ts.current("interest_rates_on_short_term_firm_loans")
        long_template = test_banks.ts.current("interest_rates_on_long_term_firm_loans")
        cons_template = test_banks.ts.current("interest_rates_on_household_consumption_loans")
        mortgage_template = test_banks.ts.current("interest_rates_on_mortgages")

        test_banks.states["Firm Short Spread"] = np.full(short_template.shape, 0.50)
        test_banks.states["Firm Long Spread"] = np.full(long_template.shape, 0.50)
        test_banks.states["Household Consumption Spread"] = np.full(cons_template.shape, 0.50)
        test_banks.states["Mortgage Spread"] = np.full(mortgage_template.shape, 0.50)

        test_banks.set_interest_rates(central_bank_policy_rate=0.01)

        np.testing.assert_allclose(test_banks.ts.current("interest_rates_on_short_term_firm_loans"), 0.04)
        np.testing.assert_allclose(test_banks.ts.current("interest_rates_on_long_term_firm_loans"), 0.05)
        np.testing.assert_allclose(test_banks.ts.current("interest_rates_on_household_consumption_loans"), 0.06)
        np.testing.assert_allclose(test_banks.ts.current("interest_rates_on_mortgages"), 0.03)
