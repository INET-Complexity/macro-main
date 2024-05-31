import numpy as np

from macromodel.individuals.individual_properties import ActivityStatus
from macromodel.timeseries import TimeSeries


def create_economy_timeseries(
    country_name: str,
    all_country_names: list[str],
    n_industries: int,
    initial_firm_prices: np.ndarray,
    initial_firm_total_sales: float,
    initial_sectoral_firm_sales: np.ndarray,
    initial_sectoral_firm_used_ii: np.ndarray,
    initial_total_taxes_on_products: float,
    initial_total_taxes_on_production: float,
    initial_change_in_firm_stock_inventories: float,
    initial_gross_fixed_capital_formation: float,
    initial_total_operating_surplus: float,
    initial_total_wages: float,
    initial_individual_activity: np.ndarray,
    initial_cpi_inflation: float,
    initial_ppi_inflation: float,
    initial_hpi_inflation: float,
    initial_real_rent_paid: np.ndarray,
    initial_imp_rent_paid: np.ndarray,
    initial_hh_rental_income: np.ndarray,
    initial_hh_consumption: float,
    initial_gov_consumption: float,
    initial_cg_rent_received: float,  # not used
    initial_cg_taxes_rental_income: float,  # not used
    initial_imports: np.ndarray,
    initial_imports_by_country: dict[str, np.ndarray],
    initial_exports: np.ndarray,
    initial_exports_by_country: dict[str, np.ndarray],
    export_taxes: float,
    initial_total_growth: float,
    initial_npl_ratio: float,
) -> TimeSeries:
    ts = TimeSeries(
        ppi=[1.0],
        cpi=[1.0],
        cfpi=[1.0],
        good_prices=np.full(n_industries, initial_firm_prices[0]),
        initial_price=[initial_firm_prices],
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
        unemployment_rate_growth=[np.nan],
        participation_rate=np.array(
            [
                (
                    np.sum(initial_individual_activity == ActivityStatus.EMPLOYED)
                    + np.sum(initial_individual_activity == ActivityStatus.UNEMPLOYED)
                )
                / len(initial_individual_activity)
            ]
        ),
        participation_rate_growth=[np.nan],
        vacancy_rate=[np.nan],
        vacancy_rate_growth=[np.nan],
        job_reallocation_rate=[np.nan],
        job_reallocation_rate_growth=[np.nan],
        #
        firm_insolvency_rate=[np.nan],
        bank_insolvency_rate=[np.nan],
        household_insolvency_rate=[np.nan],
        #
        total_growth=[initial_total_growth],
        estimated_growth=[np.nan],
        sectoral_growth=np.full(n_industries, np.nan),
        #
        hpi=[1.0],
        hpi_inflation=[initial_hpi_inflation],
        estimated_hpi_inflation=np.array([np.nan]),
        #
        total_real_rent_paid=[initial_real_rent_paid.sum()],
        total_imp_rent_paid=[initial_imp_rent_paid.sum()],
        total_real_rent_rec=[initial_hh_rental_income.sum()],
        #
        num_insolvent_firms_by_sector=np.zeros(n_industries),
        #
        npl_firm_loans=[initial_npl_ratio],
        npl_hh_cons_loans=[initial_npl_ratio],
        npl_mortgages=[initial_npl_ratio],
        #
        exports_before_taxes=initial_exports,
        exports=(1 + export_taxes) * initial_exports,
        imports=initial_imports,
        #
        gdp_output=[
            initial_firm_total_sales
            - initial_sectoral_firm_used_ii.sum()
            + initial_total_taxes_on_products
            - initial_total_taxes_on_production
            + initial_real_rent_paid.sum()
            + initial_imp_rent_paid.sum()
        ],
        gdp_output_growth=[np.nan],
        total_output=[initial_firm_total_sales],
        total_output_growth=[np.nan],
        total_intermediate_consumption=[initial_sectoral_firm_used_ii.sum()],
        total_intermediate_consumption_growth=[np.nan],
        total_gross_value_added=[initial_sectoral_firm_sales.sum() - initial_sectoral_firm_used_ii.sum()],
        total_gross_value_added_growth=[np.nan],
        total_gross_value_added_a=[initial_sectoral_firm_sales[0] - initial_sectoral_firm_used_ii[0]],
        total_gross_value_added_a_growth=[np.nan],
        total_gross_value_added_bcde=[
            initial_sectoral_firm_sales[1:5].sum() - initial_sectoral_firm_used_ii[1:5].sum()
        ],
        total_gross_value_added_bcde_growth=[np.nan],
        total_gross_value_added_c=[initial_sectoral_firm_sales[2] - initial_sectoral_firm_used_ii[2]],
        total_gross_value_added_c_growth=[np.nan],
        total_gross_value_added_f=[initial_sectoral_firm_sales[5] - initial_sectoral_firm_used_ii[5]],
        total_gross_value_added_f_growth=[np.nan],
        total_gross_value_added_ghijklmnopqrstu=[
            initial_sectoral_firm_sales[6:].sum() - initial_sectoral_firm_used_ii[6:].sum()
        ],
        total_gross_value_added_ghijklmnopqrstu_growth=[np.nan],
        total_gross_value_added_ghi=[initial_sectoral_firm_sales[6:9].sum() - initial_sectoral_firm_used_ii[6:9].sum()],
        total_gross_value_added_ghi_growth=[np.nan],
        total_gross_value_added_j=[initial_sectoral_firm_sales[9] - initial_sectoral_firm_used_ii[9]],
        total_gross_value_added_j_growth=[np.nan],
        total_gross_value_added_k=[initial_sectoral_firm_sales[10] - initial_sectoral_firm_used_ii[10]],
        total_gross_value_added_k_growth=[np.nan],
        total_gross_value_added_l=[initial_sectoral_firm_sales[11] - initial_sectoral_firm_used_ii[11]],
        total_gross_value_added_l_growth=[np.nan],
        total_gross_value_added_mn=[
            initial_sectoral_firm_sales[12:14].sum() - initial_sectoral_firm_used_ii[12:14].sum()
        ],
        total_gross_value_added_mn_growth=[np.nan],
        total_gross_value_added_opq=[
            initial_sectoral_firm_sales[14:17].sum() - initial_sectoral_firm_used_ii[14:17].sum()
        ],
        total_gross_value_added_opq_growth=[np.nan],
        total_gross_value_added_rstu=[
            initial_sectoral_firm_sales[17:].sum() - initial_sectoral_firm_used_ii[17:].sum()
        ],
        total_gross_value_added_rstu_growth=[np.nan],
        total_taxes_less_subsidies_on_products=[initial_total_taxes_on_products],
        total_taxes_less_subsidies_on_products_growth=[np.nan],
        total_taxes_on_production=[initial_total_taxes_on_production],
        total_taxes_on_production_growth=[np.nan],
        #
        # TODO: initial_gross_fixed_capital_formation is not the same as Sam's. This is because the sum of the
        #  individual incomes here is != from Sam's
        gdp_expenditure=[
            initial_change_in_firm_stock_inventories
            + initial_gross_fixed_capital_formation
            + initial_hh_consumption
            + initial_gov_consumption
            + (1 + export_taxes) * initial_exports.sum()
            - initial_imports.sum()
            + initial_real_rent_paid.sum()
            + initial_imp_rent_paid.sum()
        ],
        gdp_expenditure_growth=[np.nan],
        total_household_fce=[initial_hh_consumption],
        total_household_fce_growth=[np.nan],
        total_government_fce=[initial_gov_consumption],
        total_government_fce_growth=[np.nan],
        total_gross_fixed_capital_formation=[initial_gross_fixed_capital_formation],
        total_gross_fixed_capital_formation_growth=[np.nan],
        total_changes_in_inventories=[initial_change_in_firm_stock_inventories],
        total_changes_in_inventories_growth=[np.nan],
        #
        total_exports=[(1 + export_taxes) * initial_exports.sum()],
        total_exports_growth=[np.nan],
        #
        total_imports=[initial_imports.sum()],
        total_imports_growth=[np.nan],
        #
        gdp_income=[
            initial_total_operating_surplus
            + initial_total_wages
            + initial_total_taxes_on_products
            + initial_hh_rental_income.sum()
            + initial_cg_rent_received
            + initial_cg_taxes_rental_income
            + initial_imp_rent_paid.sum(),
        ],
        gdp_income_growth=[np.nan],
        total_gross_operating_surplus_and_mixed_income=[initial_total_operating_surplus],
        total_gross_operating_surplus_and_mixed_income_growth=[np.nan],
        total_compensation_of_employees=[initial_total_wages],
        total_compensation_of_employees_growth=[np.nan],
    )
    for c in all_country_names:
        if c == country_name:
            continue
        ts["exports_before_taxes_to_" + c] = initial_exports_by_country[c]
        ts["imports_from_" + c] = initial_imports_by_country[c]

    # GDP sanity check
    assert np.isclose(ts.current("gdp_output")[0], ts.current("gdp_expenditure")[0])
    assert np.isclose(ts.current("gdp_output")[0], ts.current("gdp_income")[0])

    return ts
