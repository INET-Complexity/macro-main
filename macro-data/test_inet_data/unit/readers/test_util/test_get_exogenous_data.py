from inet_data.readers.util.get_exogenous_data import create_all_exogenous_data


def test__create_exogenous_data(readers):
    exog_data = create_all_exogenous_data(readers, ["FRA"])
    assert exog_data["FRA"]["log_inflation"].shape[0] > 0
