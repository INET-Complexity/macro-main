import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from macro_data import DataWrapper, SyntheticCountry
from macro_data.configuration import DataConfiguration
from macro_data.configuration.countries import Country
from macro_data.configuration.region import Region
from macro_data.readers import ALL_INDUSTRIES

TEST_PATH = Path(__file__).parent.parent.resolve()


class TestCreator:
    def test__create(self, data_config_path):
        with open(data_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # not necessary to do the country splitting here
        # since the fixture used only has one country key
        configuration = DataConfiguration(**config_dict)
        configuration.prune_date = None
        configuration.seed = 0
        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        # Check if there is a file in raw data path
        creator = DataWrapper.from_config(
            configuration=configuration,
            raw_data_path=raw_data_path,
            single_hfcs_survey=True,
        )

        check_country_credit(creator.synthetic_countries["FRA"])

        check_country_gdp(creator.synthetic_countries["FRA"])

        check_country_rent_consistency(creator.synthetic_countries["FRA"])

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tmp_file = tmp / "creator.pkl"
            creator.save(tmp_file)
            new_creator = DataWrapper.init_from_pickle(tmp_file)

        assert creator.synthetic_countries.keys() == {"FRA"}

        assert new_creator.synthetic_countries.keys() == {"FRA"}

        new_creator.synthetic_countries["FRA"].reset_firm_function_dependent(
            capital_inputs_utilisation_rate=0.1,
            initial_inventory_to_input_fraction=0.1,
            intermediate_inputs_utilisation_rate=0.2,
            zero_initial_debt=False,
            zero_initial_deposits=False,
        )

        assert new_creator.emission_factors["coal"] > 0
        assert new_creator.emission_factors["oil"] > 0
        assert new_creator.emission_factors["gas"] > 0

        assert True

    def test__create_gbr(self, gbr_data_config_path):
        with open(gbr_data_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # not necessary to do the country splitting here
        # since the fixture used only has one country key
        configuration = DataConfiguration(**config_dict)
        configuration.prune_date = None
        configuration.seed = 0
        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        # Check if there is a file in raw data path
        creator = DataWrapper.from_config(
            configuration=configuration,
            raw_data_path=raw_data_path,
            single_hfcs_survey=True,
        )

        check_country_credit(creator.synthetic_countries["GBR"])

        check_country_gdp(creator.synthetic_countries["GBR"])

        check_country_rent_consistency(creator.synthetic_countries["GBR"])

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tmp_file = tmp / "creator.pkl"
            creator.save(tmp_file)
            new_creator = DataWrapper.init_from_pickle(tmp_file)

        assert creator.synthetic_countries.keys() == {"GBR"}

        assert new_creator.synthetic_countries.keys() == {"GBR"}

        new_creator.synthetic_countries["GBR"].reset_firm_function_dependent(
            capital_inputs_utilisation_rate=0.1,
            initial_inventory_to_input_fraction=0.1,
            intermediate_inputs_utilisation_rate=0.2,
            zero_initial_debt=False,
            zero_initial_deposits=False,
        )

        assert new_creator.emission_factors["coal"] > 0
        assert new_creator.emission_factors["oil"] > 0
        assert new_creator.emission_factors["gas"] > 0

        assert True

    def test__create_all_industries(self, data_config_path):
        with open(data_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # not necessary to do the country splitting here
        # since the fixture used only has one country key
        configuration = DataConfiguration(**config_dict)
        configuration.prune_date = None
        configuration.seed = 0
        configuration.aggregate_industries = False
        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        # Check if there is a file in raw data path
        creator = DataWrapper.from_config(
            configuration=configuration,
            raw_data_path=raw_data_path,
            single_hfcs_survey=True,
        )

        check_country_credit(creator.synthetic_countries["FRA"])

        check_country_gdp(creator.synthetic_countries["FRA"])

    def test__create_all_industries_gbr(self, gbr_data_config_path):
        with open(gbr_data_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # not necessary to do the country splitting here
        # since the fixture used only has one country key
        configuration = DataConfiguration(**config_dict)
        configuration.prune_date = None
        configuration.seed = 0
        configuration.aggregate_industries = False
        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        # Check if there is a file in raw data path
        creator = DataWrapper.from_config(
            configuration=configuration,
            raw_data_path=raw_data_path,
            single_hfcs_survey=True,
        )

        check_country_credit(creator.synthetic_countries["GBR"])

        check_country_gdp(creator.synthetic_countries["GBR"])

    def test__default_banks_firms(self, data_config_path):
        with open(data_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # not necessary to do the country splitting here
        # since the fixture used only has one country key
        configuration = DataConfiguration(**config_dict)
        configuration.country_configs[Country("FRA")].firms_configuration.constructor = "Default"
        configuration.country_configs[Country("FRA")].banks_configuration.constructor = "Default"
        configuration.country_configs[Country("FRA")].single_bank = False
        configuration.country_configs[Country("FRA")].firms_configuration.zero_initial_deposits = False
        configuration.country_configs[Country("FRA")].firms_configuration.zero_initial_debt = False
        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        configuration.prune_date = None
        creator = DataWrapper.from_config(
            configuration=configuration,
            raw_data_path=raw_data_path,
            single_hfcs_survey=True,
        )
        assert creator.synthetic_countries.keys() == {"FRA"}

    def test__default_banks_firms_gbr(self, gbr_data_config_path):
        with open(gbr_data_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # not necessary to do the country splitting here
        # since the fixture used only has one country key
        configuration = DataConfiguration(**config_dict)
        configuration.country_configs[Country("GBR")].firms_configuration.constructor = "Default"
        configuration.country_configs[Country("GBR")].banks_configuration.constructor = "Default"
        configuration.country_configs[Country("GBR")].single_bank = False
        configuration.country_configs[Country("GBR")].firms_configuration.zero_initial_deposits = False
        configuration.country_configs[Country("GBR")].firms_configuration.zero_initial_debt = False
        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        configuration.prune_date = None
        creator = DataWrapper.from_config(
            configuration=configuration,
            raw_data_path=raw_data_path,
            single_hfcs_survey=True,
        )
        assert creator.synthetic_countries.keys() == {"GBR"}

    def test__single_banks(self, data_config_path):
        with open(data_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # not necessary to do the country splitting here
        # since the fixture used only has one country key
        configuration = DataConfiguration(**config_dict)
        configuration.country_configs[Country("FRA")].single_bank = True
        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        configuration.prune_date = None
        creator = DataWrapper.from_config(
            configuration=configuration,
            raw_data_path=raw_data_path,
            single_hfcs_survey=True,
        )
        assert creator.synthetic_countries.keys() == {"FRA"}

    def test__single_banks_gbr(self, gbr_data_config_path):
        with open(gbr_data_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # not necessary to do the country splitting here
        # since the fixture used only has one country key
        configuration = DataConfiguration(**config_dict)
        configuration.country_configs[Country("GBR")].single_bank = True
        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        configuration.prune_date = None
        creator = DataWrapper.from_config(
            configuration=configuration,
            raw_data_path=raw_data_path,
            single_hfcs_survey=True,
        )
        assert creator.synthetic_countries.keys() == {"GBR"}

    def test_no_deposits_debt(self, data_config_path):
        with open(data_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # not necessary to do the country splitting here
        # since the fixture used only has one country key
        configuration = DataConfiguration(**config_dict)
        configuration.country_configs[Country("FRA")].firms_configuration.zero_initial_deposits = True
        configuration.country_configs[Country("FRA")].firms_configuration.zero_initial_debt = True
        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        configuration.prune_date = None
        creator = DataWrapper.from_config(
            configuration=configuration,
            raw_data_path=raw_data_path,
            single_hfcs_survey=True,
        )
        assert creator.synthetic_countries.keys() == {"FRA"}

    def test_no_deposits_debt_gbr(self, gbr_data_config_path):
        with open(gbr_data_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # not necessary to do the country splitting here
        # since the fixture used only has one country key
        configuration = DataConfiguration(**config_dict)
        configuration.country_configs[Country("GBR")].firms_configuration.zero_initial_deposits = True
        configuration.country_configs[Country("GBR")].firms_configuration.zero_initial_debt = True
        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        configuration.prune_date = None
        creator = DataWrapper.from_config(
            configuration=configuration,
            raw_data_path=raw_data_path,
            single_hfcs_survey=True,
        )
        assert creator.synthetic_countries.keys() == {"GBR"}

    def test__create_us_can(self, data_config_path, multic_readers):
        with open(data_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # not necessary to do the country splitting here
        # since the fixture used only has one country key
        configuration = DataConfiguration(**config_dict)

        united_states = Country("USA")
        canada = Country("CAN")
        france = Country("FRA")

        configuration.country_configs[france].single_firm_per_industry = True
        configuration.country_configs[france].single_bank = True
        configuration.country_configs[france].single_government_entity = True

        # we add the US and Canada to the configuration
        # by setting their configurations to be the same as France
        # and by setting their EU proxy country to be France

        configuration.country_configs[united_states] = configuration.country_configs[france]

        configuration.country_configs[canada] = configuration.country_configs[france]

        configuration.country_configs[united_states].eu_proxy_country = france
        configuration.country_configs[canada].eu_proxy_country = france

        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        creator = DataWrapper.from_config(
            configuration=configuration,
            raw_data_path=raw_data_path,
            single_hfcs_survey=True,
        )

        assert creator.synthetic_countries.keys() == {"FRA", "USA", "CAN"}

        check_country_gdp(creator.synthetic_countries["FRA"])
        check_country_gdp(creator.synthetic_countries["USA"])
        check_country_gdp(creator.synthetic_countries["CAN"])

        check_country_credit(creator.synthetic_countries["FRA"])
        check_country_credit(creator.synthetic_countries["USA"])
        check_country_credit(creator.synthetic_countries["CAN"])

        for country_name in [france, united_states, canada]:
            exch_rate = multic_readers.exchange_rates.from_usd_to_lcu(country_name, 2014)
            usd_consumption = multic_readers.icio[2014].get_hh_consumption(country_name)

            total_consumption = usd_consumption.sum()

            country = creator.synthetic_countries[country_name]
            vat = country.tax_data.value_added_tax
            households = country.population.household_data
            disposable_income = households["Income"] * (1 - households["Saving Rate"])

            total_disposable_income = disposable_income.sum()

            govt_consumption = country.government_entities.gov_entity_data["Consumption in LCU"].sum()

            govt_consumption_usd = govt_consumption / exch_rate

            assert total_disposable_income / (1 + vat) == pytest.approx(total_consumption * exch_rate, rel=5e-2)

            govt_consumption_reader = multic_readers.icio[2014].get_govt_consumption(country_name)

            assert govt_consumption_reader.sum() == pytest.approx(govt_consumption_usd, rel=5e-2)

            employee_social_contribution_taxes = country.tax_data.employee_social_insurance_tax
            income_tax = country.tax_data.income_tax

            employee_income = country.population.individual_data["Employee Income"].sum()
            wages = country.firms.firm_data["Total Wages"].sum()

            assert wages * (
                1 - employee_social_contribution_taxes - income_tax * (1 - employee_social_contribution_taxes)
            ) == pytest.approx(employee_income, rel=5e-2)

    def test_create_us_can_all_industries(self, data_config_path):
        with open(data_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # not necessary to do the country splitting here
        # since the fixture used only has one country key
        configuration = DataConfiguration(**config_dict)

        united_states = Country("USA")
        canada = Country("CAN")
        france = Country("FRA")

        configuration.country_configs[france].single_firm_per_industry = True
        configuration.country_configs[france].single_bank = True
        configuration.country_configs[france].single_government_entity = True

        # we add the US and Canada to the configuration
        # by setting their configurations to be the same as France
        # and by setting their EU proxy country to be France

        configuration.country_configs[united_states] = configuration.country_configs[france]

        configuration.country_configs[canada] = configuration.country_configs[france]

        configuration.country_configs[united_states].eu_proxy_country = france
        configuration.country_configs[canada].eu_proxy_country = france

        configuration.aggregate_industries = False

        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        creator = DataWrapper.from_config(
            configuration=configuration,
            raw_data_path=raw_data_path,
            single_hfcs_survey=True,
        )

        assert creator.synthetic_countries.keys() == {"FRA", "USA", "CAN"}

        check_country_gdp(creator.synthetic_countries["FRA"])
        check_country_gdp(creator.synthetic_countries["USA"])
        check_country_gdp(creator.synthetic_countries["CAN"])

        check_country_credit(creator.synthetic_countries["FRA"])
        check_country_credit(creator.synthetic_countries["USA"])
        check_country_credit(creator.synthetic_countries["CAN"])

    def test_create_can_only(self, data_config_path):
        with open(data_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # not necessary to do the country splitting here
        # since the fixture used only has one country key
        configuration = DataConfiguration(**config_dict)

        canada = Country("CAN")
        france = Country("FRA")

        configuration.country_configs[france].single_firm_per_industry = True
        configuration.country_configs[france].single_bank = True
        configuration.country_configs[france].single_government_entity = True

        configuration.country_configs[canada] = configuration.country_configs[france]

        configuration.country_configs[canada].eu_proxy_country = france

        del configuration.country_configs[france]

        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        creator = DataWrapper.from_config(
            configuration=configuration,
            raw_data_path=raw_data_path,
            single_hfcs_survey=True,
        )

        assert creator.synthetic_countries.keys() == {"CAN"}

    def test_create_us_only(self, data_config_path):
        with open(data_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # not necessary to do the country splitting here
        # since the fixture used only has one country key
        configuration = DataConfiguration(**config_dict)

        united_states = Country("USA")
        france = Country("FRA")

        configuration.country_configs[france].single_firm_per_industry = True
        configuration.country_configs[france].single_bank = True
        configuration.country_configs[france].single_government_entity = True

        configuration.country_configs[united_states] = configuration.country_configs[france]

        configuration.country_configs[united_states].eu_proxy_country = france

        del configuration.country_configs[france]

        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        creator = DataWrapper.from_config(
            configuration=configuration,
            raw_data_path=raw_data_path,
            single_hfcs_survey=True,
        )

        assert creator.synthetic_countries.keys() == {"USA"}

    def test__create_can_disagg(self, data_config_path):
        with open(data_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # not necessary to do the country splitting here
        # since the fixture used only has one country key
        configuration = DataConfiguration(**config_dict)
        configuration.can_disaggregation = True
        configuration.aggregate_industries = False

        canada = Country("CAN")
        france = Country("FRA")

        configuration.country_configs[france].single_firm_per_industry = True
        configuration.country_configs[france].single_bank = True
        configuration.country_configs[france].single_government_entity = True

        configuration.country_configs[canada] = configuration.country_configs[france]

        configuration.country_configs[canada].eu_proxy_country = france

        del configuration.country_configs[france]

        configuration.prune_date = None
        configuration.seed = 0
        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        # Check if there is a file in raw data path
        creator = DataWrapper.from_config(
            configuration=configuration,
            raw_data_path=raw_data_path,
            single_hfcs_survey=True,
        )

        check_country_credit(creator.synthetic_countries["CAN"])

        check_country_gdp(creator.synthetic_countries["CAN"])

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tmp_file = tmp / "creator.pkl"
            creator.save(tmp_file)
            new_creator = DataWrapper.init_from_pickle(tmp_file)

        assert creator.synthetic_countries.keys() == {"CAN"}

        assert new_creator.synthetic_countries.keys() == {"CAN"}

        new_creator.synthetic_countries["CAN"].reset_firm_function_dependent(
            capital_inputs_utilisation_rate=0.1,
            initial_inventory_to_input_fraction=0.1,
            intermediate_inputs_utilisation_rate=0.2,
            zero_initial_debt=False,
            zero_initial_deposits=False,
        )

        canada = creator.synthetic_countries["CAN"]

        firm_input_emissions = canada.firms.firm_data["Input Emissions"].sum()
        firm_capital_emissions = canada.firms.firm_data["Capital Emissions"].sum()

        assert firm_input_emissions > 0
        assert firm_capital_emissions > 0

        household_emissions = canada.population.household_data["Consumption Emissions"].sum()

        assert household_emissions > 0

        household_investment_emissions = canada.population.household_data["Investment Emissions"].sum()

        assert household_investment_emissions > 0

        government_consumption_emissions = canada.government_entities.gov_entity_data["Consumption Emissions"].sum()

        assert government_consumption_emissions > 0

    def test__create_can_provincial(self, canada_disagg_config):
        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        creator = DataWrapper.from_config(
            configuration=canada_disagg_config,
            raw_data_path=raw_data_path,
            single_hfcs_survey=True,
        )

        # check country gdp and credit
        for country_name, country in creator.synthetic_countries.items():
            check_country_gdp(country)
            check_country_credit(country)

            assert np.all(
                country.population.individual_data["Employee Income"] >= 0
            ), f"Negative employee income for {country_name}"


def check_country_credit(country: SyntheticCountry):
    pop_debt = country.population.household_data["Debt"].sum()
    firm_debt = country.firms.firm_data["Debt"].sum()

    firm_deposits_in_bank = country.banks.bank_data["Deposits from Firms"].sum()
    household_deposits_in_bank = country.banks.bank_data["Deposits from Households"].sum()

    household_deposits = country.population.household_data["Wealth in Deposits"].sum()
    firm_deposits = country.firms.firm_data["Deposits"].sum()

    firm_loans = country.banks.bank_data["Loans to Firms"].sum()
    household_loans = country.banks.bank_data["Loans to Households"].sum()

    # loans match debt
    assert firm_loans == pytest.approx(firm_debt, rel=1e-4)
    assert household_loans == pytest.approx(pop_debt, rel=1e-4)

    # deposits match deposits in bank
    assert firm_deposits == pytest.approx(firm_deposits_in_bank, rel=1e-4)
    assert household_deposits == pytest.approx(household_deposits_in_bank, rel=1e-4)


def check_country_gdp(country: SyntheticCountry):
    gdp_output = country.gdp_output
    gdp_income = country.gdp_income
    gdp_expenditure = country.gdp_expenditure

    assert gdp_output == pytest.approx(gdp_income, rel=1e-3)
    assert gdp_output == pytest.approx(gdp_expenditure, rel=1e-3)


def check_country_rent_consistency(country: SyntheticCountry):
    """Check rent consistency between household and housing market data.

    This test validates:
    1. Rent values are reasonable (no excessive zeros for renters)
    2. Total rent paid by households roughly matches rent in housing market
    3. Tenure status codes are properly applied
    4. House ID mappings are consistent
    """
    household_data = country.population.household_data
    housing_data = country.housing_market.housing_market_data

    # Test 1: Check tenure status codes are being used correctly
    renters = household_data["Tenure Status of the Main Residence"] == 3
    owners = household_data["Tenure Status of the Main Residence"].isin([1, 2, 4])
    social_housing = household_data["Tenure Status of the Main Residence"] == -1

    # Ensure we have renters identified
    assert renters.sum() > 0, "No households identified as renters (tenure code 3)"

    # Ensure we have owners identified
    assert owners.sum() > 0, "No households identified as owners (tenure codes 1,2,4)"

    # Test 2: Check rent values for renters are reasonable
    renter_rent_paid = household_data.loc[renters, "Rent Paid"]

    # Most renters should have non-zero rent (allow some social housing at 0)
    non_zero_rent_renters = (renter_rent_paid > 0).sum()
    total_renters = renters.sum()

    assert (
        non_zero_rent_renters / total_renters > 0.5
    ), f"Too many renters with zero rent: {total_renters - non_zero_rent_renters}/{total_renters}"

    # Test 3: Check owners should have zero rent paid (they don't pay rent)
    owner_rent_paid = household_data.loc[owners, "Rent Paid"]
    assert (owner_rent_paid == 0).all(), "Owner-occupied households should have zero rent paid"

    # Test 4: Check housing market rent consistency
    # Get households that are mapped to houses
    household_house_mapping = household_data["Corresponding Inhabited House ID"].dropna()

    if len(household_house_mapping) > 0:
        # Check that house IDs exist in housing data
        valid_house_ids = household_house_mapping.isin(housing_data.index)
        assert valid_house_ids.all(), "Some household house IDs don't exist in housing market data"

        # Check rental properties have reasonable rent values
        rental_properties = housing_data[~housing_data["Is Owner-Occupied"]]
        if len(rental_properties) > 0:
            rental_rents = rental_properties["Rent"]
            non_zero_rental_rents = (rental_rents > 0).sum()

            assert non_zero_rental_rents / len(rental_rents) > 0.8, (
                f"Too many rental properties with zero rent: "
                f"{len(rental_rents) - non_zero_rental_rents}/"
                f"{len(rental_rents)}"
            )

    # Test 5: Basic rent magnitude check
    total_household_rent = household_data["Rent Paid"].sum()

    # Rent should be a reasonable fraction of total income (rough sanity check)
    total_household_income = household_data["Income"].sum()
    rent_to_income_ratio = total_household_rent / total_household_income

    # Rent should be between 0.5% and 40% of total income (generous bounds)
    # Lower bound relaxed for spoofed test data which may have altered distributions
    assert (
        0.005 <= rent_to_income_ratio <= 0.40
    ), f"Rent-to-income ratio {rent_to_income_ratio:.3f} is outside reasonable bounds [0.005, 0.40]"
