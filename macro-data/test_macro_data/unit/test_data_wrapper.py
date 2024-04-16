import pytest
import yaml

from macro_data import DataWrapper
from pathlib import Path
import tempfile

from macro_data.configuration import DataConfiguration
from macro_data.configuration.countries import Country

TEST_PATH = Path(__file__).parent.parent.resolve()


class TestCreator:
    def test__create(self, data_config_path):
        with open(data_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # not necessary to do the country splitting here
        # since the fixture used only has one country key
        configuration = DataConfiguration(**config_dict)
        configuration.prune_date = None
        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        creator = DataWrapper.from_config(
            configuration=configuration,
            raw_data_path=raw_data_path,
            single_hfcs_survey=True,
        )

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

        assert True

    def test__default_banks_firms(self, data_config_path):
        with open(data_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # not necessary to do the country splitting here
        # since the fixture used only has one country key
        configuration = DataConfiguration(**config_dict)
        configuration.country_configs[Country("FRA")].firms_configuration.constructor = "Default"
        configuration.country_configs[Country("FRA")].banks_configuration.constructor = "Default"
        raw_data_path = TEST_PATH / "unit" / "sample_raw_data"
        configuration.prune_date = None
        creator = DataWrapper.from_config(
            configuration=configuration,
            raw_data_path=raw_data_path,
            single_hfcs_survey=True,
        )
        assert creator.synthetic_countries.keys() == {"FRA"}

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
