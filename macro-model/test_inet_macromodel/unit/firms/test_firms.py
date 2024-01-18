import numpy as np

from configurations import FirmsConfiguration
from firms import Firms


class TestFirms:
    def test__create(self, test_firms):
        assert test_firms.country_name == "FRA"

    def test__firms_states(self, test_firms):
        assert test_firms is not None
        for state in [
            "Industry",
            "Corresponding Bank ID",
            "Employments",
            "is_insolvent",
            "Excess Demand",
        ]:
            assert state in test_firms.states.keys()

    def test__firms_ts(self, test_firms):
        for ts_key in [
            "n_firms",
            "n_firms_by_industry",
            "number_of_employees",
            "production",
            "unconstrained_target_production",
            "constrained_target_production",
            "target_production",
            "price",
            "price_in_usd",
            "profits",
            "taxes_paid_on_production",
            "corporate_taxes_paid",
            "equity",
            "estimated_demand",
            "demand",
            "unconstrained_target_intermediate_inputs",
            "unconstrained_target_intermediate_inputs_costs",
            "unconstrained_target_capital_inputs",
            "unconstrained_target_capital_inputs_costs",
            "target_intermediate_inputs",
            "target_capital_inputs",
            "inventory",
            "intermediate_inputs_stock",
            "intermediate_inputs_stock_value",
            "used_intermediate_inputs",
            "used_intermediate_inputs_costs",
            "capital_inputs_stock",
            "capital_inputs_stock_value",
            "used_capital_inputs",
            "used_capital_inputs_costs",
            "real_amount_bought_as_intermediate_inputs",
            "real_amount_bought_as_capital_goods",
            "total_sales",
            "target_short_term_credit",
            "target_long_term_credit",
            "received_short_term_credit",
            "received_long_term_credit",
            "received_credit",
            "short_term_loan_debt",
            "long_term_loan_debt",
            "debt",
            "debt_installments",
            "interest_paid_on_deposits",
            "interest_paid_on_loans",
            "interest_paid",
            "deposits",
            "estimated_growth_by_firm",
            "labour_inputs",
            "desired_labour_inputs",
            "labour_costs",
            "real_amount_bought_as_capital_inputs",
        ]:
            assert ts_key in test_firms.ts.get_keys()

    def test__compute_labour_inputs(self, test_firms):
        assert np.allclose(
            test_firms.compute_labour_inputs(
                corresponding_firm=np.arange(18),
                current_labour_inputs=np.full(18, 1.0),
            ),
            np.full(18, 1.0),
        )

    def test__compute_n_employees(self, test_firms):
        assert np.allclose(test_firms.compute_n_employees(corresponding_firm=np.arange(18)), np.full(18, 1))

    def test__compute_total_wages_paid(self, test_firms):
        assert np.allclose(
            test_firms.compute_total_wages_paid(
                corresponding_firm=np.arange(18),
                individual_wages=np.full(18, 2.0),
                income_taxes=0.2,
                employee_social_insurance_tax=0.05,
                employer_social_insurance_tax=0.03,
            ),
            np.full(18, 2.71052632),
        )

    def test__compute_inventory(self, test_firms):
        test_firms.ts.real_amount_sold.append(np.full(18, 0.5))
        inv1 = test_firms.compute_inventory()
        test_firms.ts.real_amount_sold.append(inv1)
        assert np.allclose(test_firms.compute_inventory(), np.full(18, 0.5))

    def test__compute_intermediate_inputs_stock(self, test_firms):
        test_firms.ts.used_intermediate_inputs.append(np.full((18, 18), 1))
        test_firms.ts.real_amount_bought_as_intermediate_inputs.append(np.zeros((18, 18)))
        stock1 = test_firms.compute_intermediate_inputs_stock()
        test_firms.ts.used_intermediate_inputs.append(stock1)
        assert np.allclose(
            test_firms.compute_intermediate_inputs_stock().astype(float),
            np.full((18, 18), 1.0),
        )

    def test__compute_capital_inputs_stock(self, test_firms):
        stock1 = test_firms.ts.current("capital_inputs_stock")
        test_firms.ts.used_capital_inputs.append(stock1)
        assert np.allclose(
            test_firms.compute_capital_inputs_stock().astype(float),
            np.diag(np.full(18, 0.0)),
        )

    def test__compute_profits(self, test_firms):
        test_firms.ts.price.append(np.full(18, 1.0))
        test_firms.ts.production.append(np.full(18, 1.0))
        test_firms.ts.used_intermediate_inputs_costs.append(np.full(18, 10.0))
        test_firms.ts.used_capital_inputs_costs.append(np.full(18, 10.0))
        test_firms.ts.taxes_paid_on_production.append(np.full(18, 1.0))
        test_firms.ts.interest_paid.append(np.full(18, 2.0))
        test_firms.ts["interest_received"] = np.full(18, 3.0)
        test_firms.ts.total_wage.append(np.full(18, 1.0))
        assert np.allclose(
            test_firms.compute_profits(),
            np.full(18, -23),
        )

    def test__compute_deposits(self, test_firms):
        pass

    def test__compute_debt(self, test_firms):
        test_firms.ts.debt.append(np.full(18, 10.0))
        test_firms.ts.debt_installments.append(np.full(18, 0.5))
        test_firms.ts.received_credit.append(np.full(18, 3.0))
        test_firms.ts.short_term_loan_debt.append(np.full(18, 3.0))
        test_firms.ts.long_term_loan_debt.append(np.full(18, 10.0))
        assert np.allclose(test_firms.compute_debt(), np.full(18, 13.0))

    # def test__compute_equity(self, test_firms):
    #     test_firms.ts.current("intermediate_inputs_stock")
    #     assert np.allclose(
    #         test_firms.compute_equity(
    #             current_good_prices=np.full(18, 1.0),
    #         ),
    #         np.full(18, -8.5),
    #     )
