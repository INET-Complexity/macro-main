import pathlib
import numpy as np

from inet_data.processing.synthetic_population.hfcs_synthetic_population import (
    SyntheticHFCSPopulation,
)

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


class TestSyntheticPopulation:
    def test__create(self, readers):
        population = SyntheticHFCSPopulation(
            country_name="FRA",
            country_name_short="FR",
            scale=10000,
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
        population.create(
            hfcs_reader=readers["hfcs"]["FRA"],
            econ_reader=readers["oecd_econ"],
            wb_reader=readers["world_bank"],
            n_households=1000,
            number_of_firms_by_industry=np.ones(18).astype(int),
            total_unemployment_benefits=10000.0,
            rent_as_fraction_of_unemployment_rate=0.3,
        )
        population.set_consumption_weights(
            consumption_weights=np.full(18, 1.0 / 18),
        )
        population.compute_household_wealth(wealth_distribution_independents=["Income"])
        population.compute_household_income(
            central_gov_config={
                "functions": {
                    "household_social_transfers": {
                        "parameters": {
                            "independents": {"value": ["Wealth"]},
                        }
                    }
                }
            },
            total_social_transfers=1000.0,
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
