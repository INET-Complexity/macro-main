import numpy as np

from central_government import CentralGovernment
from configurations import CentralGovernmentConfiguration
from inet_macromodel.individuals.individual_properties import ActivityStatus


class TestCentralGovernment:
    def test__create(self, datawrapper, test_individuals):
        country = datawrapper.synthetic_countries["FRA"]
        synthetic_central_government = country.central_government

        central_government_config = CentralGovernmentConfiguration()

        taxes_less_subsidies = country.industry_data["industry_vectors"]["Taxes Less Subsidies Rates"].values

        n_industries = len(datawrapper.configuration.industries)

        n_unemployed = np.sum(test_individuals.states["Activity Status"] == ActivityStatus.UNEMPLOYED)

        central_government = CentralGovernment.from_pickled_agent(
            synthetic_central_government=synthetic_central_government,
            configuration=central_government_config,
            country_name="FRA",
            all_country_names=["FRA"],
            taxes_net_subsidies=taxes_less_subsidies,
            tax_data=country.tax_data,
            n_industries=n_industries,
            number_of_unemployed_individuals=n_unemployed,
        )

        assert central_government.country_name == "FRA"

    def test__central_government_states(self, test_central_government):
        assert test_central_government is not None
        for state in [
            "Value-added Tax",
            "Export Tax",
            "Employer Social Insurance Tax",
            "Employee Social Insurance Tax",
            "Profit Tax",
            "Income Tax",
            "Taxes Less Subsidies Rates",
        ]:
            assert state in test_central_government.states.keys()

    def test__central_government_ts(self, test_central_government):
        for ts_key in [
            "unemployment_benefits_by_individual",
            "total_other_benefits",
        ]:
            assert ts_key in test_central_government.ts.get_keys()

    def test__distribute_unemployment_benefits_to_individuals(self, test_central_government):
        assert np.allclose(
            test_central_government.distribute_unemployment_benefits_to_individuals(
                current_individual_activity_status=np.array([ActivityStatus.EMPLOYED, ActivityStatus.UNEMPLOYED]),
            ),
            np.array([0.0, 1800.0]),
        )

    def test__compute_taxes_revenue_deficit_debt(self, test_central_government):
        test_central_government.compute_taxes(
            current_ind_employee_income=np.array([50.0, 100.0]),
            current_total_rent_paid=np.array([10.0, 30.0]),
            current_income_financial_assets=np.array([5.0, 5.0]),
            current_ind_activity=np.array([ActivityStatus.EMPLOYED, ActivityStatus.UNEMPLOYED]),
            current_ind_realised_cons=np.array([50.0, 100.0]),
            current_bank_profits=np.array([10.0]),
            current_firm_production=np.array([200.0]),
            current_firm_price=np.array([1.0]),
            current_firm_profits=np.array([20.0]),
            current_firm_industries=np.array([0]),
            current_household_new_real_wealth=np.array([15.0]),
            taxes_less_subsidies_rates=np.array([0.2]),
            current_total_exports=100.0,
        )
        test_central_government.ts["debt"] = np.array([50.0])
        test_central_government.ts["revenue"] = np.array([40.0])
        assert test_central_government.compute_revenue(household_rent_paid_to_government=100.0) == 140.0
