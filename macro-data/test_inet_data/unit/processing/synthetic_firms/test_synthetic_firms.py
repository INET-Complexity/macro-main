import pathlib

import numpy as np
import pandas as pd

from inet_data.processing.synthetic_firms.default_synthetic_firms import (
    SyntheticDefaultFirms,
)

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve()


class TestSyntheticFirms:
    def test__create(self, readers, industry_data):
        firms = SyntheticDefaultFirms(
            country_name="FRA",
            scale=100000,
            year=2014,
            industries=[
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "K",
                "L",
                "M",
                "N",
                "O",
                "P",
                "Q",
                "R_S",
            ],
        )
        firms.set_industries(number_of_firms_by_industry=np.ones(18).astype(int))
        firms.create(
            econ_reader=readers.oecd_econ,
            ons_reader=readers.ons,
            exchange_rates=readers.exchange_rates,
            total_firm_deposits=10000,
            total_firm_debt=0,
            industry_data=industry_data["FRA"],
            number_of_employees_by_industry=100 * np.ones(18).astype(int),
            intermediate_inputs_utilisation_rate=0.9,
            capital_inputs_utilisation_rate=0.9,
            assume_zero_initial_deposits=True,
            assume_zero_initial_debt=True,
            initial_inventory_to_production_fraction=0.0,
        )
        firms.firm_data["Corresponding Bank ID"] = 0
        firms.set_additional_initial_conditions(
            econ_reader=readers.oecd_econ,
            industry_data=industry_data["FRA"],
            interest_rate_on_firm_deposits=np.full(18, 0.02),
            overdraft_rate_on_firm_deposits=np.full(18, 0.03),
            credit_market_data=pd.DataFrame({"loan_type": [1], "loan_recipient_id": [0]}),
        )

        # Check if we have all the necessary fields
        for firm_field in [
            "Industry",
            "Number of Employees",
            "Total Wages",
            "Production",
            "Price in USD",
            "Price",
            "Demand",
            "Labour Inputs",
            "Inventory",
            "Deposits",
            "Debt",
            "Equity",
            "Corresponding Bank ID",
            "Taxes paid on Production",
            "Interest paid on deposits",
            "Interest paid on loans",
            "Interest paid",
            "Profits",
            "Corporate Taxes Paid",
            "Debt Installments",
        ]:
            assert firm_field in firms.firm_data.columns

        # Check if there are any missing values
        assert not np.any(pd.isna(firms.firm_data))
