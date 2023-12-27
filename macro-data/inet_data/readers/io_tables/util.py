import pandas as pd


def aggregate_df(aggregation: dict, country_agg_dict: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate a DataFrame based on given aggregation and country aggregation dictionaries.

    Countries can be aggregated together (i.e. mapping multiple countries to the ROW, or mapping regions to a country).

    Sectors are also aggregated together (i.e. mapping multiple sectors to a single sector).

    Args:
        aggregation (dict): A dictionary mapping aggregated values to their corresponding keys.
        country_agg_dict (dict): A dictionary mapping country indices to their corresponding keys.
        df (pd.DataFrame): The DataFrame to be aggregated.

    Returns:
        pd.DataFrame: The aggregated DataFrame.

    """
    agg_dict_full = {}
    for key, values in aggregation.items():
        for value in values:
            agg_dict_full[value] = key
    stacked = df.stack().stack().reset_index().rename(columns={0: "Value"})
    stacked["NewCountryInd"] = stacked["CountryInd"].map(country_agg_dict)
    stacked["NewindustryInd"] = stacked["industryInd"].map(agg_dict_full)
    stacked["NewindustryCol"] = stacked["industryCol"].map(agg_dict_full)
    stacked["NewCountryCol"] = stacked["CountryCol"].map(country_agg_dict)
    aggregated: pd.DataFrame = (
        stacked.groupby(["NewCountryInd", "NewindustryInd", "NewindustryCol", "NewCountryCol"])["Value"]
        .sum()
        .unstack()
        .unstack()
    )
    return aggregated
