from inet_data.readers.util.prune_util import filter_columns_by_date
import pandas as pd


def test_filter_columns_by_date():
    columns = ["name", "description", "2020-01-01", "2021-01-01", "2019-01-01"]
    date = "2020-01-01"
    # empty dataframe with same columns
    df = pd.DataFrame(columns=columns)

    result = filter_columns_by_date(df.columns, date)
    assert result == ["name", "description", "2020-01-01", "2021-01-01"]

    date = 2020
    df = pd.DataFrame(columns=columns)

    result = filter_columns_by_date(df.columns, date)
    assert result == ["name", "description", "2020-01-01", "2021-01-01"]

    date = pd.to_datetime("January 1, 2020")

    result = filter_columns_by_date(df.columns, date)
    assert result == ["name", "description", "2020-01-01", "2021-01-01"]
