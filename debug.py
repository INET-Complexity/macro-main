from macro_data import DataWrapper
from macro_data.configuration_utils import default_data_configuration

DATA_PATH = "/Users/jmoran/Projects/macrocosm/inet/data/raw_data"


def create_pickle(configuration, filename):
    creator = DataWrapper.from_config(configuration=configuration, raw_data_path=DATA_PATH, single_hfcs_survey=False)

    creator.save(filename)


if __name__ == "__main__":
    # configuration for France, Canada, and the USA
    can_usa_fra_configuration = default_data_configuration(
        countries=["FRA", "CAN", "USA"],
        proxy_country_dict={"CAN": "FRA", "USA": "FRA"},
        aggregate_industries=False,
    )

    # configuration for France, Canada, and the USA
    can_fra_configuration = default_data_configuration(
        countries=["FRA", "CAN"],
        proxy_country_dict={"CAN": "FRA"},
        aggregate_industries=False,
    )

    # configuration for France, Canada, and the USA
    fra_configuration = default_data_configuration(countries=["FRA"], aggregate_industries=False)
    print("Creating pickle for FRA...")
    create_pickle(fra_configuration, "fra_data.pkl")
