from typing import Optional

import pandas as pd
from sklearn.impute import IterativeImputer  # noqa


def apply_iterative_imputer(
    df: pd.DataFrame, columns: list[str], selection: Optional[pd.Series] = None, **imputer_args
):
    imputer = IterativeImputer(**imputer_args)
    if selection is None:
        df.loc[:, columns] = imputer.fit_transform(df[columns].values)
        return df
    else:
        df.loc[selection, columns] = imputer.fit_transform(df.loc[selection, columns].values)
        return df
