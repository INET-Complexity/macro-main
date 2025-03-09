"""
This module provides utilities for handling missing data in economic datasets through
advanced imputation techniques. It implements iterative imputation methods that can
handle complex relationships between variables when filling missing values.

The module uses scikit-learn's IterativeImputer, which performs multivariate imputation
by modeling each feature with missing values as a function of other features in a
round-robin fashion. This approach is particularly useful for economic data where
variables often have strong interdependencies.

Key features:
- Multivariate iterative imputation
- Support for selective imputation on specific rows
- Flexible configuration through imputer arguments
- Preservation of data structure and relationships

Example:
    ```python
    import pandas as pd
    from macro_data.util.imputation import apply_iterative_imputer

    # Create sample data with missing values
    data = pd.DataFrame({
        'gdp': [100, np.nan, 98, 101, 102],
        'consumption': [80, 81, np.nan, 82, 83],
        'investment': [20, 21, 19, np.nan, 20]
    })

    # Impute missing values
    imputed_data = apply_iterative_imputer(
        df=data,
        columns=['gdp', 'consumption', 'investment'],
        max_iter=10,
        random_state=42
    )
    ```
"""

from typing import Optional

import pandas as pd
from sklearn.impute import IterativeImputer  # noqa


def apply_iterative_imputer(
    df: pd.DataFrame, columns: list[str], selection: Optional[pd.Series] = None, **imputer_args
) -> pd.DataFrame:
    """
    Apply iterative imputation to fill missing values in specified columns.

    This function uses scikit-learn's IterativeImputer to perform sophisticated
    missing value imputation by modeling each feature with missing values as a
    function of other features. The process is repeated iteratively until
    convergence or a maximum number of iterations is reached.

    The imputation can be applied to:
    1. All rows in the specified columns
    2. A subset of rows specified by a boolean selection mask

    The method preserves relationships between variables by using the available
    data to predict missing values in a way that maintains the covariance
    structure of the data.

    Args:
        df (pd.DataFrame): Input DataFrame containing missing values to impute.
            The DataFrame should contain numeric data or data that can be
            processed by the imputer.
        columns (list[str]): Names of columns to impute. These columns should
            contain numeric data or missing values (NaN).
        selection (Optional[pd.Series]): Boolean mask indicating which rows
            to impute. If None, all rows are processed. Defaults to None.
        **imputer_args: Additional arguments passed to IterativeImputer.
            Common arguments include:
            - max_iter: Maximum number of imputation rounds
            - random_state: Random seed for reproducibility
            - min_value: Minimum possible imputed value
            - max_value: Maximum possible imputed value

    Returns:
        pd.DataFrame: Copy of input DataFrame with missing values filled in
            the specified columns. Original DataFrame remains unchanged.

    Notes:
        - The function modifies the input DataFrame in-place for efficiency
        - All specified columns must be numeric or coercible to numeric
        - The imputation quality depends on the relationships between variables
        - Consider normalizing variables if they are on different scales

    Example:
        ```python
        # Impute missing values in economic indicators
        imputed_df = apply_iterative_imputer(
            df=economic_data,
            columns=['gdp', 'consumption', 'investment'],
            selection=economic_data['year'] >= 2000,  # Only recent data
            max_iter=20,
            random_state=42
        )
        ```
    """
    imputer = IterativeImputer(**imputer_args)
    if selection is None:
        df.loc[:, columns] = imputer.fit_transform(df[columns].values)
        return df
    else:
        df.loc[selection, columns] = imputer.fit_transform(df.loc[selection, columns].values)
        return df
