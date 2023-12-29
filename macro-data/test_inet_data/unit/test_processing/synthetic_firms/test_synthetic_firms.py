import pathlib

import numpy as np
import pandas as pd

from inet_data.processing.synthetic_firms.default_synthetic_firms import (
    DefaultSyntheticFirms,
)

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve()


class TestSyntheticFirms:
    def test__create(self, readers, industry_data):
        industries = [
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
        ]

        n_employees_per_industry = np.ones(18).astype(int)
        n_employees_per_industry *= 10_000

        firms = DefaultSyntheticFirms.from_readers(
            readers=readers,
            country_name="FRA",
            year=2014,
            industries=industries,
            industry_data=industry_data["FRA"],
            n_employees_per_industry=n_employees_per_industry,
            assume_zero_initial_debt=False,
            assume_zero_initial_deposits=False,
            scale=10000,
        )

        # firms.firm_data["Corresponding Bank ID"] = 0
        # firms.set_additional_initial_conditions(
        #     econ_reader=readers.oecd_econ,
        #     industry_data=industry_data["FRA"],
        #     interest_rate_on_firm_deposits=np.full(18, 0.02),
        #     overdraft_rate_on_firm_deposits=np.full(18, 0.03),
        #     credit_market_data=pd.DataFrame({"loan_type": [1], "loan_recipient_id": [0]}),
        # )

        # These are all the fields firms have
        # but not all that are created at initialization
        # all_fields = [
        #     "Industry",
        #     "Number of Employees",
        #     "Total Wages",
        #     "Production",
        #     "Price in USD",
        #     "Price",
        #     "Demand",
        #     "Labour Inputs",
        #     "Inventory",
        #     "Deposits",
        #     "Debt",
        #     "Equity",
        #     "Corresponding Bank ID",
        #     "Taxes paid on Production",
        #     "Interest paid on deposits",
        #     "Interest paid on loans",
        #     "Interest paid",
        #     "Profits",
        #     "Corporate Taxes Paid",
        #     "Debt Installments",
        # ]

        # Check if we have all the necessary fields
        init_fields = [
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
        ]
        assert set(init_fields).issubset(firms.firm_data.columns)
        # Check if there are any missing values
        assert not np.any(pd.isna(firms.firm_data))
