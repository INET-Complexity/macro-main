import pathlib
import numpy as np
import pandas as pd

from data.processing.synthetic_central_government.default_synthetic_central_government import (
    SyntheticDefaultCentralGovernment,
)

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


class TestSyntheticCentralGovernment:
    def test__create(self, readers):
        central_gov = SyntheticDefaultCentralGovernment(
            country_name="FRA",
            year=2014,
        )
        central_gov.create(
            central_gov_debt=readers["oecd_econ"].general_gov_debt("FRA", 2014),
            benefits_data=pd.DataFrame(
                data={"Unemployment Benefits": [100.0, 100.0], "Other Total Benefits": [200.0, 200.0]},
                index=pd.DatetimeIndex(pd.date_range(start="2010-01-01", end="2012-01-01", freq="Y")),
            ),
            exogenous_data={
                "log_inflation": pd.DataFrame(
                    {"Real CPI Inflation": [0.05] * 24},
                    index=[
                        "2010-01",
                        "2010-02",
                        "2010-03",
                        "2010-04",
                        "2010-05",
                        "2010-06",
                        "2010-07",
                        "2010-08",
                        "2010-09",
                        "2010-10",
                        "2010-11",
                        "2010-12",
                        "2011-01",
                        "2011-02",
                        "2011-03",
                        "2011-04",
                        "2011-05",
                        "2011-06",
                        "2011-07",
                        "2011-08",
                        "2011-09",
                        "2011-10",
                        "2011-11",
                        "2011-12",
                    ],
                ),
                "unemployment_rate": pd.DataFrame(
                    {"Unemployment Rate": [0.2] * 24},
                    index=[
                        "2010-01",
                        "2010-02",
                        "2010-03",
                        "2010-04",
                        "2010-05",
                        "2010-06",
                        "2010-07",
                        "2010-08",
                        "2010-09",
                        "2010-10",
                        "2010-11",
                        "2010-12",
                        "2011-01",
                        "2011-02",
                        "2011-03",
                        "2011-04",
                        "2011-05",
                        "2011-06",
                        "2011-07",
                        "2011-08",
                        "2011-09",
                        "2011-10",
                        "2011-11",
                        "2011-12",
                    ],
                ),
            },
            regression_window=12,
        )

        # Check if we have all the necessary fields
        for central_gov_field in [
            "Debt",
            "Total Unemployment Benefits",
            "Other Social Benefits",
        ]:
            assert central_gov_field in central_gov.central_gov_data.columns

        # Check if there are any missing values
        assert not np.any(pd.isna(central_gov.central_gov_data))
