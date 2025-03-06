from macro_data.configuration_utils import default_data_configuration


class TestConfigUtils:
    def test__create_fra(self):
        data_conf = default_data_configuration(countries=["FRA"])
        assert True

    def test__create_can_error(self):
        try:
            data_conf = default_data_configuration(countries=["CAN"])
            assert False
        except ValueError:
            assert True

        try:
            data_conf = default_data_configuration(countries=["CAN"], proxy_country_dict={"CAN": "USA"})
            assert False
        except ValueError:
            assert True

    def test__create_can(self):
        data_conf = default_data_configuration(countries=["CAN"], proxy_country_dict={"CAN": "FRA"})
        assert True
