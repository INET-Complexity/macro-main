from macro_data.configuration_utils import default_data_configuration


class TestConfigUtils:
    def test__create_fra(self):
        default_data_configuration(countries=["FRA"])
        assert True

    def test__create_can_error(self):
        try:
            default_data_configuration(countries=["CAN"])
            assert False
        except ValueError:
            assert True

        try:
            default_data_configuration(countries=["CAN"], proxy_country_dict={"CAN": "USA"})
            assert False
        except ValueError:
            assert True

    def test__create_can(self):
        default_data_configuration(countries=["CAN"], proxy_country_dict={"CAN": "FRA"})
        assert True
