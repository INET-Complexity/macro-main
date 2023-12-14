import pathlib

import numpy as np

from inet_data.processing.synthetic_population.hfcs_synthetic_population import (
    SyntheticHFCSPopulation,
)

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


class TestSyntheticPopulation:
    def test__init(self, readers, configuration, industry_data):
        industries = configuration["model"]["industries"]["value"]

        population = SyntheticHFCSPopulation.create_from_readers(
            readers=readers,
            country_name="FRA",
            year=2014,
            scale=10000,
            country_name_short="FR",
            industries=industries,
            industry_data=industry_data,
            rent_as_fraction_of_unemployment_rate=0.5,
            total_unemployment_benefits=1000.0,
        )

        # Check if we have all the necessary fields
        for ind_field in [
            "Gender",
            "Age",
            "Education",
            "Activity Status",
            "Employment Industry",
            "Employee Income",
            "Income from Unemployment Benefits",
            "Income",
            "Corresponding Household ID",
        ]:
            assert ind_field in population.individual_data.columns
        for hh_field in [
            "Tenure Status of the Main Residence",
            "Rent Paid",
            "Number of Properties other than Household Main Residence",
            "Type",
            "Rental Income from Real Estate",
            "Income from Pensions",
            "Regular Social Transfers",
            "Value of the Main Residence",
            "Value of other Properties",
            "Value of Household Vehicles",
            "Value of Household Valuables",
            "Value of Self-Employment Businesses",
            "Wealth in Deposits",
            "Mutual Funds",
            "Bonds",
            "Value of Private Businesses",
            "Shares",
            "Managed Accounts",
            "Money owed to Households",
            "Other Assets",
            "Voluntary Pension",
            "Outstanding Balance of HMR Mortgages",
            "Outstanding Balance of Mortgages on other Properties",
            "Outstanding Balance of Credit Line",
            "Outstanding Balance of Credit Card Debt",
            "Outstanding Balance of other Non-Mortgage Loans",
            "Consumption of Consumer Goods/Services as a Share of Income",
            "Corresponding Individuals ID",
            "Wealth Other Real Assets",
            "Wealth in Real Assets",
            "Wealth in Other Financial Assets",
            "Wealth in Financial Assets",
            "Wealth",
            "Debt",
            "Net Wealth",
            "Employee Income",
            "Income",
        ]:
            assert hh_field in population.household_data.columns

        # Check individual gender
        assert np.all(population.individual_data["Gender"].isin([1, 2]))

        # Check individual age
        assert np.all(population.individual_data["Age"] >= 0)
