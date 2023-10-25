"""
Test for the data pruning functions.
"""
import pandas as pd

from inet_data.readers.handle_readers import (
    filter_columns_by_date,
    prune_eurostat,
    prune_icio,
    prune_imf,
    prune_oecd,
    prune_policy_rates,
    prune_wb_exchange_rates,
    prune_wiod_sea,
    prune_world_bank,
)


# Mock data
class MockWorldBank:
    """
    A dictionary where each key has a DataFrame as its value. Each DataFrame can have either 'year' or 'time' as a column or years as columns.
    """

    def __init__(self):
        self.data = {
            "dataset_1": pd.DataFrame({"year": ["2019", "2020", "2021", "2022"]}),
            "dataset_2": pd.DataFrame({"2020-01": [1, 2], "2021-01": [3, 4], "2022-01": [5, 6]}),
        }


class MockWIODSEA:
    """An object that has an attribute 'exchange_rates' which has a DataFrame as an attribute."""

    def __init__(self):
        self.exchange_rates = lambda: None
        self.exchange_rates.df = pd.DataFrame({"2020-01": [1, 2], "2021-01": [3, 4], "2022-01": [5, 6]})


class MockIMF:
    """
    An object that has an attribute 'data' which is a dictionary containing DataFrames
    """

    def __init__(self):
        self.data = {"bank_demography": pd.DataFrame({"2020-01": [1], "2021-01": [2], "2022-01": [3]})}


class MockPolicyRates:
    """
    An object with an attribute 'df' which is a DataFrame.
    """

    def __init__(self):
        self.df = pd.DataFrame({"2020-01": [0.5], "2021-01": [0.75], "2022-01": [1.0]})


class MockOECD:
    def __init__(self):
        self.data = {
            "dataset_1": pd.DataFrame({"year": ["2019", "2020", "2021", "2022"]}),
            "dataset_2": pd.DataFrame({"country_year": ["US_2019", "US_2020", "US_2021", "US_2022"]}),
        }


def MockICIO():
    return {"2019": "data_2019", "2020": "data_2020", "2021": "data_2021", "2022": "data_2022"}


class MockEurostat:
    def __init__(self):
        self.data = {
            "dataset_1": pd.DataFrame({"TIME_PERIOD": ["2019", "2020", "2021", "2022"]}),
            "dataset_2": pd.DataFrame({"2020-01": [10, 20], "2021-01": [30, 40], "2022-01": [50, 60]}),
        }


class MockWBExchangeRates(MockPolicyRates):  # Reuse the same structure
    pass


# Test for filter_columns_by_date
def test_filter_columns_by_date():
    columns = ["name", "description", "2020-01-01", "2021-01-01", "2019-01-01"]
    date = "2020-01-01"
    result = filter_columns_by_date(columns, date)
    assert result == ["name", "description", "2020-01-01", "2021-01-01"]


# Test for prune_world_bank
def test_prune_world_bank():
    wb = MockWorldBank()
    start_date = "2021"
    pruned_wb = prune_world_bank(wb, start_date)
    assert "2019" not in pruned_wb.data["dataset_1"]["year"].values
    assert "2020-01" not in pruned_wb.data["dataset_2"].columns


# Test for prune_wiod_sea
def test_prune_wiod_sea():
    wiod = MockWIODSEA()
    start_date = "2021"
    pruned_wiod = prune_wiod_sea(wiod, start_date)
    assert "2020-01" not in pruned_wiod.exchange_rates.df.columns


# Test for prune_imf
def test_prune_imf():
    imf = MockIMF()
    start_date = "2021"
    pruned_imf = prune_imf(imf, start_date)
    assert "2020-01" not in pruned_imf.data["bank_demography"].columns


# Test for prune_policy_rates
def test_prune_policy_rates():
    rates = MockPolicyRates()
    start_date = "2021"
    pruned_rates = prune_policy_rates(rates, start_date)
    assert "2020-01" not in pruned_rates.df.columns


# Test for prune_oecd
def test_prune_oecd():
    oecd = MockOECD()
    start_date = "2021"
    pruned_oecd = prune_oecd(oecd, start_date)
    assert "2019" not in pruned_oecd.data["dataset_1"]["year"].values
    assert "US_2019" not in pruned_oecd.data["dataset_2"]["country_year"].values


# Test for prune_icio
def test_prune_icio():
    icio = MockICIO()
    start_date = "2021"
    pruned_icio = prune_icio(icio, start_date)
    assert "2019" not in pruned_icio


# Test for prune_eurostat
def test_prune_eurostat():
    eurostat = MockEurostat()
    start_date = "2021"
    pruned_eurostat = prune_eurostat(eurostat, start_date)
    assert "2019" not in pruned_eurostat.data["dataset_1"]["TIME_PERIOD"].values
    assert "2020-01" not in pruned_eurostat.data["dataset_2"].columns


# Test for prune_wb_exchange_rates
def test_prune_wb_exchange_rates():
    rates = MockWBExchangeRates()
    start_date = "2021"
    pruned_rates = prune_wb_exchange_rates(rates, start_date)
    assert "2020-01" not in pruned_rates.df.columns
