from typing import Optional

import numpy as np
import pandas as pd
import cvxpy as cp

from macro_data.configuration.countries import Country
from macro_data.configuration.region import Region
from macro_data.readers.economic_data.eurostat_reader import EuroStatReader
from macro_data.readers.io_tables.icio_reader import ICIOReader
from macro_data.readers.socioeconomic_data.wiod_sea_data import WIODSEAReader


def match_iot_with_sea(
    icio_reader: ICIOReader,
    sea_reader: WIODSEAReader,
    country_names: list[str | Country | Region],
    yearly_factor: float = 4.0,
    regions_dict: Optional[dict[Country, list[Region]]] = None,
) -> None:
    for country_name in country_names:
        if regions_dict is None:
            _match_country_iot_with_sea(country_name, icio_reader, sea_reader, yearly_factor)
        else:
            if country_name in regions_dict:
                for region in regions_dict[country_name]:
                    _match_country_iot_with_sea(region, icio_reader, sea_reader, yearly_factor)
            else:
                _match_country_iot_with_sea(country_name, icio_reader, sea_reader, yearly_factor)


def add_investment_matrix_to_icio(
    icio_reader: ICIOReader,
    sea_reader: WIODSEAReader,
    country_names: list[str | Country],
    yearly_factor: float = 4.0,
    regions_dict: Optional[dict[Country, list[Region]]] = None,
) -> None:
    for country_name in country_names:
        if regions_dict is None:
            _add_country_investment(country_name, icio_reader, sea_reader, yearly_factor)
        else:
            # check if country is in regions_dict
            if country_name in regions_dict:
                for region in regions_dict[country_name]:
                    _add_country_investment(region, icio_reader, sea_reader, yearly_factor)
            else:
                _add_country_investment(country_name, icio_reader, sea_reader, yearly_factor)


def _add_country_investment(
    country_name: Country | Region, icio_reader: ICIOReader, sea_reader: WIODSEAReader, yearly_factor: float = 4.0
):
    gfcf = icio_reader.column_allc(country_name, "Firm Fixed Capital Formation") / icio_reader.yearly_factor
    cap_factors = sea_reader.get_values_in_usd(country_name, "Capital Compensation") / gfcf
    value_added = icio_reader.get_value_added(country_name) * yearly_factor

    # replace nans with 0
    cap_factors = np.where(np.isnan(cap_factors), 0, cap_factors)
    cap_factors /= cap_factors.sum()  # normalise to 1

    violated_constraint = cap_factors >= icio_reader.get_value_added(country_name) / gfcf.sum()
    if np.any(violated_constraint):
        ratios = sea_reader.get_values_in_usd(country_name, "Capital Compensation") / value_added
        ratios = np.where(np.isnan(ratios), 0, ratios)
        max_capital_ratio = ratios.max()
        cap_factors = adjust_c_vector(
            c_vector=cap_factors,
            v=icio_reader.get_value_added(country_name),
            g=gfcf,
            gamma=max_capital_ratio,
        )
    #     cap_factors[violated_constraint] = (
    #         0.5 * np.mean(cap_factors[~violated_constraint])
    #         + 0.5 * sea_reader.get_values_in_usd(country_name, "Value Added")[violated_constraint] / gfcf.sum()
    #     )
    #     cap_factors /= cap_factors.sum()  # normalise to 1

    investment_matrix = np.array([gfcf for _ in range(len(cap_factors))]).T
    investment_matrix = np.einsum("ij, j-> ij", investment_matrix, cap_factors)  # proportionally fitting CAP

    assert np.allclose(investment_matrix.sum(axis=1), gfcf, rtol=1e-3)

    capital_ratios = np.divide(
        investment_matrix.sum(axis=0),
        sea_reader.get_values_in_usd(country_name, "Value Added"),
        out=np.zeros_like(gfcf),
        where=value_added != 0,
    )

    assert np.all(capital_ratios <= 1.0), f"Capital ratios for {country_name} exceed 1.0: {capital_ratios}"

    # investment_matrix *= 1 / np.sum(cap_factors)  # match GFCF exactly
    investment_matrix = (
        pd.DataFrame(
            data=investment_matrix,
            index=pd.MultiIndex.from_product(
                [[country_name], icio_reader.industries],
                names=["Country", "Industry"],
            ),
            columns=pd.MultiIndex.from_product(
                [[country_name], icio_reader.industries],
                names=["Country", "Industry"],
            ),
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    icio_reader.investment_matrices[country_name] = investment_matrix


def get_sea(
    country_name: str,
    field: str,
    sea_reader: WIODSEAReader,
) -> np.ndarray:
    return sea_reader.df.loc[
        sea_reader.df.index.get_level_values(0) == country_name,
        field,
    ].values


def _match_country_iot_with_sea(
    country_name: Country | Region, icio_reader: ICIOReader, sea_reader: WIODSEAReader, yearly_factor: float = 4.0
):
    # sea_reader.df.loc[
    #     sea_reader.df.index.get_level_values(0) == country_name,
    #     "Capital Compensation",
    # ] = yearly_factor * icio_reader.investment_matrices[country_name].values.sum(axis=0)
    sea_reader.set_values_in_usd(
        country_name,
        "Capital Compensation",
        yearly_factor * icio_reader.investment_matrices[country_name].values.sum(axis=0),
    )
    new_va = yearly_factor * icio_reader.get_value_added(country_name)
    va_factor = new_va / get_sea(country_name, "Value Added", sea_reader)
    # sea_reader.df.loc[
    #     sea_reader.df.index.get_level_values(0) == country_name,
    #     "Value Added",
    # ] = new_va
    sea_reader.set_values_in_usd(country_name, "Value Added", new_va)
    # sea_reader.df.loc[
    #     sea_reader.df.index.get_level_values(0) == country_name,
    #     "Labour Compensation",
    # ] = get_sea(
    #     country_name, "Value Added", sea_reader
    # ) - get_sea(country_name, "Capital Compensation", sea_reader)
    sea_reader.set_values_in_usd(
        country_name,
        "Labour Compensation",
        get_sea(country_name, "Value Added", sea_reader) - get_sea(country_name, "Capital Compensation", sea_reader),
    )
    # sea_reader.df.loc[
    #     sea_reader.df.index.get_level_values(0) == country_name,
    #     "Capital Stock",
    # ] *= va_factor
    sea_reader.df.loc[country_name].loc[sea_reader.industries, "Capital Stock"] *= va_factor

    sea_reader.df.loc[sea_reader.df["Value Added"] == 0] = 0


def reconcile_value_added(
    icio_reader: ICIOReader,
    sea_reader: WIODSEAReader,
    country_names: list[str | Country | Region],
    yearly_factor: float = 4.0,
    regions_dict: Optional[dict[Country, list[Region]]] = None,
) -> None:
    for country_name in country_names:
        if regions_dict is None:
            _reconcile_value_added(country_name, icio_reader, sea_reader, yearly_factor)
        else:
            if country_name in regions_dict:
                for region in regions_dict[country_name]:
                    _reconcile_value_added(region, icio_reader, sea_reader, yearly_factor)
            else:
                _reconcile_value_added(country_name, icio_reader, sea_reader, yearly_factor)


def _reconcile_value_added(
    country_name: Country | Region, icio_reader: ICIOReader, sea_reader: WIODSEAReader, yearly_factor: float = 4.0
):
    new_va = yearly_factor * icio_reader.get_value_added_series(country_name)
    old_va = sea_reader.df.loc[country_name, "Value Added"]

    va_factor = new_va.loc[old_va.index] / old_va

    va_factor = va_factor.values
    va_factor = np.where(np.isnan(va_factor), 0, va_factor)

    sea_reader.df.loc[country_name, "Value Added"] = new_va.loc[old_va.index].values

    sea_reader.df.loc[country_name].loc[sea_reader.industries, "Labour Compensation"] *= va_factor

    sea_reader.df.loc[country_name].loc[sea_reader.industries, "Capital Compensation"] *= va_factor

    sea_reader.df.loc[country_name].loc[sea_reader.industries, "Capital Stock"] *= va_factor * va_factor


def _match_country_iot_with_sea2(
    country_name: Country | Region, icio_reader: ICIOReader, sea_reader: WIODSEAReader, yearly_factor: float = 4.0
):
    sea_reader.df.loc[
        sea_reader.df.index.get_level_values(0) == country_name,
        "Capital Compensation",
    ] = yearly_factor * icio_reader.investment_matrices[country_name].values.sum(axis=0)
    new_va = yearly_factor * icio_reader.get_value_added(country_name)
    va_factor = new_va / get_sea(country_name, "Value Added", sea_reader)
    sea_reader.df.loc[
        sea_reader.df.index.get_level_values(0) == country_name,
        "Value Added",
    ] = new_va

    # labour compensation + capital compensation = value added, but we need to offset by a factor

    sea_value_added = (
        sea_reader.df.loc[sea_reader.df.index.get_level_values(0) == country_name, "Labour Compensation"]
        + sea_reader.df.loc[sea_reader.df.index.get_level_values(0) == country_name, "Capital Compensation"]
    )

    ratio = new_va / sea_value_added

    sea_reader.df.loc[
        sea_reader.df.index.get_level_values(0) == country_name,
        "Labour Compensation",
    ] *= ratio

    sea_reader.df.loc[
        sea_reader.df.index.get_level_values(0) == country_name,
        "Capital Compensation",
    ] *= ratio

    sea_reader.df.loc[
        sea_reader.df.index.get_level_values(0) == country_name,
        "Capital Stock",
    ] *= (
        va_factor * ratio
    )

    # icio_capital_columns = icio_reader.iot.columns.get_level_values(1).str.contains("Capital Formation")
    #
    # icio_reader.iot.loc[country_name, icio_capital_columns] *= ratio


def get_investment_fractions(
    country_names: list[Country | Region],
    eurostat: EuroStatReader,
    proxy_country_dict: dict[Country, Country],
    year: int,
) -> dict[Country, dict[str, float]]:
    investment_fractions = {}
    for country_name in country_names:
        data_country = country_name
        if isinstance(country_name, Region):
            data_country = country_name.parent_country
        if not data_country.is_eu_country:
            data_country = proxy_country_dict[data_country]

        investment_fractions[country_name] = eurostat.get_investment_fractions_of_country(data_country, year=year)
    return investment_fractions


def adjust_c_vector(c_vector: np.ndarray, g: np.ndarray, v: np.ndarray, gamma=0.99):
    """
    Adjust vector C to a new vector C' that is close to C and satisfies
      C'[j] <= gamma * (v[j]/sum(g)) * sum(C')
    for each j.

    Parameters:
      c_vector: Original vector (numpy array) of size m.
      g: Vector (numpy array) for capital formation (size n).
      v: Vector (numpy array) for the upper bounds (size m).
      gamma: Slack factor (<1 to avoid saturation).

    Returns:
      C_prime: Adjusted vector (numpy array) of size m.
    """
    m = len(c_vector)
    # Define variable for C' and the sum S = sum(C')
    C_prime = cp.Variable(m, nonneg=True)
    S = cp.Variable(nonneg=True)

    # A_j = v[j] / sum(g)
    A = v / np.sum(g)

    # Define constraints:
    constraints = []
    constraints.append(cp.sum(C_prime) == S)
    for j in range(m):
        constraints.append(C_prime[j] <= gamma * A[j] * S)

    # Objective: minimize squared distance to original C
    objective = cp.Minimize(cp.sum_squares(C_prime - c_vector))

    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError("Optimization did not converge")

    return C_prime.value / S.value
