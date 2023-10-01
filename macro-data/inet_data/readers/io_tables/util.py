import pandas as pd


def aggregate_df(aggregation: dict, country_agg_dict: dict, df: pd.DataFrame):
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
