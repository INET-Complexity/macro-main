from inet_data.readers.default_readers import prune_icio_dict
from inet_data.readers.util.get_exogenous_data import create_all_exogenous_data


def test__prune_icio_dict():
    dummy_dict = {
        2014: "dummy",
        2015: "dummy",
        2016: "dummy",
        2017: "dummy",
        2018: "dummy",
        2019: "dummy",
        2020: "dummy",
    }

    pruned_dict = prune_icio_dict(dummy_dict, 2016)
    assert pruned_dict == {2016: "dummy", 2017: "dummy", 2018: "dummy", 2019: "dummy", 2020: "dummy"}


def test__get_benefits_inflation(readers):
    exogenous_data = create_all_exogenous_data(readers, ["FRA"])
    data = readers.get_benefits_inflation_data("FRA", 2004, 2014, exogenous_data["FRA"])
    assert data.shape[0] > 0
