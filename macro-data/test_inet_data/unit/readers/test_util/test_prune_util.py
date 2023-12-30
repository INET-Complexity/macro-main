from inet_data.readers.util.prune_util import prune_index
import pandas as pd


def test_prune_index():
    columns = ["name", "description", "2020-01-01", "2021-01-01", "2019-01-01"]
    prune_date = pd.to_datetime("2020-01-01").date()
    # convert date to date
    # empty dataframe with same columns
    df = pd.DataFrame(columns=columns)

    result = prune_index(df.columns, prune_date)
    assert result == ["name", "description", "2020-01-01", "2021-01-01"]

    prune_date = pd.to_datetime("January 1, 2020").date()

    result = prune_index(df.columns, prune_date)
    assert result == ["name", "description", "2020-01-01", "2021-01-01"]
