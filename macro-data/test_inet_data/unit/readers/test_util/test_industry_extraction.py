from inet_data.readers.util.industry_extraction import compile_exogenous_industry_data


def test__compile_exogenous_industry_data(readers):
    country_names = ["FRA"]
    industry_data = compile_exogenous_industry_data(readers, country_names)
    assert industry_data["FRA"].shape[0] > 0
