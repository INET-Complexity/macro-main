import pathlib

import numpy as np
import pytest

from macro_data.configuration.countries import Country
from macro_data.processing.synthetic_population.hfcs_synthetic_population import (
    SyntheticHFCSPopulation,
    sample_households,
)

PARENT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


class TestSyntheticPopulation:
    def test__init(self, readers, configuration, industry_data):
        industries = configuration.industries

        france = Country("FRA")

        population = SyntheticHFCSPopulation.from_readers(
            readers=readers,
            country_name=france,
            year=2014,
            scale=10000,
            country_name_short=france.to_two_letter_code(),
            industries=industries,
            industry_data=industry_data[france],
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


@pytest.mark.parametrize("country", [Country("CAN"), Country("USA")])
def test__household_consumption(multic_readers, multic_industry_data, configuration, country):
    industries = configuration.industries

    proxy_country = Country("FRA")
    year = 2014
    population_ratio = multic_readers.world_bank.get_population(
        country=country, year=year
    ) / multic_readers.world_bank.get_population(country=proxy_country, year=year)

    exch_rate_proxy_to_lcu = multic_readers.exchange_rates.from_eur_to_lcu(country, year)

    population = SyntheticHFCSPopulation.from_readers(
        readers=multic_readers,
        country_name=proxy_country,
        year=2014,
        scale=10000,
        country_name_short=proxy_country.to_two_letter_code(),
        industries=industries,
        industry_data=multic_industry_data[country],
        rent_as_fraction_of_unemployment_rate=0.5,
        total_unemployment_benefits=1000.0,
        population_ratio=population_ratio,
        exch_rate=exch_rate_proxy_to_lcu,
    )

    assert True


def test__household_sampling(readers):
    country_name = "FRA"
    year = 2014
    scale = 10_000
    n_households = int(readers.eurostat.number_of_households(country_name, year) / scale)
    hfcs_individuals_data = readers.hfcs[country_name].individuals_df
    hfcs_households_data = readers.hfcs[country_name].households_df

    household_selection, individual_selection = sample_households(
        hfcs_households_data, hfcs_individuals_data, n_households
    )

    assert household_selection.shape[0] == n_households
    assert individual_selection["New Household ID"].nunique() == n_households
    assert np.all(
        individual_selection.groupby(["New Household ID"])["Gender"].count()
        == household_selection["Corresponding Individuals ID"].apply(len)
    )

    large_households = household_selection["Corresponding Individuals ID"].apply(len) > 2

    sample = np.random.choice(household_selection.index[large_households], size=10)

    for i in sample:
        individuals = household_selection.loc[i, "Corresponding Individuals ID"]
        assert individual_selection.loc[individuals, "Corresponding Household ID"].nunique() == 1
        assert np.all(individual_selection.loc[individuals, "Corresponding Household ID"] == i)
    assert True
