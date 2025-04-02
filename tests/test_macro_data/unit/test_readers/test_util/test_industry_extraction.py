def test__compile_exogenous_industry_data(exogenous_industry_data):
    assert exogenous_industry_data["FRA"].shape[0] > 0


def test__industry_data(industry_data):
    row_exports = industry_data["ROW"]["industry_vectors"]["Exports in USD to FRA"]
    assert row_exports.sum() > 0
