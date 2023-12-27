from typing import Optional

import pandas as pd
from sklearn.impute import IterativeImputer  # noqa


def apply_iterative_imputer(
    df: pd.DataFrame, columns: list[str], selection: Optional[pd.Series] = None, **imputer_args
) -> pd.DataFrame:
    """
    Apply iterative imputation to fill missing values in the specified columns of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be imputed.
        columns (list[str]): The list of column names to be imputed.
        selection (Optional[pd.Series]): An optional boolean series indicating the rows to be imputed.
        **imputer_args: Additional arguments to be passed to the IterativeImputer.

    Returns:
        pd.DataFrame: The DataFrame with missing values imputed.

    """
    imputer = IterativeImputer(**imputer_args)
    if selection is None:
        df.loc[:, columns] = imputer.fit_transform(df[columns].values)
        return df
    else:
        df.loc[selection, columns] = imputer.fit_transform(df.loc[selection, columns].values)
        return df
