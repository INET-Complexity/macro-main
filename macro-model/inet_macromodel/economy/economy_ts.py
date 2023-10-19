import numpy as np

from inet_macromodel.timeseries import TimeSeries
from inet_macromodel.individuals.individual_properties import ActivityStatus


def create_economy_timeseries(
    country_name: str,
    all_country_names: list[str],
    n_industries: int,
    initial_firm_prices: float,
    initial_firm_total_sales: float,
    initial_firm_total_used_ii: float,
    initial_total_taxes_on_products: float,
    initial_change_in_firm_stock_inventories: float,
    initial_total_operating_surplus_plus_wages: float,
    initial_individual_activity: np.ndarray,
    initial_cpi_inflation: float,
    initial_ppi_inflation: float,
    initial_nominal_house_price_index_growth: float,
    initial_real_rent_paid: np.ndarray,
    initial_imp_rent_paid: np.ndarray,
    initial_hh_rental_income: np.ndarray,
    initial_hh_consumption: float,
    initial_gov_consumption: float,
    initial_cg_rent_received: float,
    initial_cg_taxes_rental_income: float,
    initial_sectoral_growth: np.ndarray,
    initial_sentiment: float,
    initial_imports: np.ndarray,
    initial_imports_by_country: dict[str, np.ndarray],
    initial_exports: np.ndarray,
    initial_exports_by_country: dict[str, np.ndarray],
    export_taxes: float,
) -> TimeSeries:
    ts = TimeSeries(
        ppi=[initial_firm_prices],
        cpi=[initial_firm_prices],
        cfpi=[initial_firm_prices],
        good_prices=np.full(n_industries, initial_firm_prices),
        #
        cpi_inflation=[initial_cpi_inflation],
        ppi_inflation=[initial_ppi_inflation],
        cfpi_inflation=[np.nan],
        industry_inflation=np.full(n_industries, np.nan),
        estimated_cpi_inflation=[np.nan],
        estimated_ppi_inflation=[np.nan],
        #
        unemployment_rate=np.array(
            [
                np.sum(initial_individual_activity == ActivityStatus.UNEMPLOYED)
                / (
                    np.sum(initial_individual_activity == ActivityStatus.EMPLOYED)
                    + np.sum(initial_individual_activity == ActivityStatus.UNEMPLOYED)
                )
            ]
        ),
        participation_rate=np.array(
            [
                (
                    np.sum(initial_individual_activity == ActivityStatus.EMPLOYED)
                    + np.sum(initial_individual_activity == ActivityStatus.UNEMPLOYED)
                )
                / len(initial_individual_activity)
            ]
        ),
        vacancy_rate=[0.0],
        job_reallocation_rate=[0.0],
        #
        firm_insolvency_rate=[0.0],
        bank_insolvency_rate=[0.0],
        household_insolvency_rate=[0.0],
        #
        total_growth=[np.nan],
        sectoral_growth=initial_sectoral_growth,
        estimated_sectoral_growth=np.full(n_industries, np.nan),
        #
        nominal_house_price_index_growth=[initial_nominal_house_price_index_growth],
        estimated_nominal_house_price_index_growth=np.array([np.nan]),
        #
        total_real_rent_paid=[initial_real_rent_paid.sum()],
        total_imp_rent_paid=[initial_imp_rent_paid.sum()],
        total_real_rent_rec=[initial_hh_rental_income.sum()],
        #
        sectoral_sentiment=np.full(n_industries, initial_sentiment),
        #
        num_insolvent_firms_by_sector=np.zeros(n_industries),
        #
        exports_before_taxes=initial_exports,
        exports=(1 + export_taxes) * initial_exports,
        imports=initial_imports,
        #
        gdp_output=[
            initial_firm_total_sales
            - initial_firm_total_used_ii
            + initial_total_taxes_on_products
            + initial_real_rent_paid.sum()
            + initial_imp_rent_paid.sum()
        ],
        gdp_expenditure=[
            initial_change_in_firm_stock_inventories
            + initial_hh_consumption
            + initial_gov_consumption
            + (1 + export_taxes) * initial_exports.sum()
            - initial_imports.sum()
            + initial_real_rent_paid.sum()
            + initial_imp_rent_paid.sum()
        ],
        gdp_income=[
            initial_total_operating_surplus_plus_wages
            + initial_total_taxes_on_products
            + initial_hh_rental_income.sum()
            + initial_cg_rent_received
            + initial_cg_taxes_rental_income
            + initial_imp_rent_paid.sum()
        ],
    )
    for c in all_country_names:
        if c == country_name:
            continue
        ts["exports_before_taxes_to_" + c] = initial_exports_by_country[c]
        ts["imports_from_" + c] = initial_imports_by_country[c]

    return ts
