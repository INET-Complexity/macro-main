import numpy as np
import pandas as pd

from pathlib import Path

from inet_data import SyntheticCountry

from inet_macromodel.firms.firms import Firms
from inet_macromodel.timeseries import TimeSeries
from inet_macromodel.households.households import Households
from inet_macromodel.util.function_mapping import get_functions
from inet_macromodel.economy.economy_ts import create_economy_timeseries
from inet_macromodel.individuals.individual_properties import ActivityStatus
from inet_macromodel.government_entities.government_entities import GovernmentEntities

from typing import Any, Optional


class Economy:
    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        functions: dict[str, Any],
        ts: TimeSeries,
        states: dict[str, float | np.ndarray | list[np.ndarray]],
    ):
        self.country_name = country_name
        self.all_country_names = all_country_names
        self.n_industries = n_industries
        self.functions = functions
        self.n_industries = n_industries
        self.ts = ts
        self.states = states

    @classmethod
    def from_agents(
        cls,
        country_name: str,
        all_country_names: list[str],
    ):
        ...

    @classmethod
    def from_data(
        cls,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        initial_firm_prices: np.ndarray,
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
        initial_imports: np.ndarray,
        initial_imports_by_country: dict[str, np.ndarray],
        initial_exports: np.ndarray,
        initial_exports_by_country: dict[str, np.ndarray],
        export_taxes: float,
        config: dict[str, Any],
    ) -> "Economy":
        # Get corresponding functions
        functions = get_functions(
            config["functions"],
            loc="inet_macromodel.economy",
            func_dir=Path(__file__).parent / "func",
        )

        # Take fixed parameters
        if "parameters" in config.keys():
            parameters = config["parameters"].copy()
        else:
            parameters = {}
        parameters["n_industries"] = n_industries

        # Additional states
        states: dict[str, float | np.ndarray | list[np.ndarray]] = {}

        # Create the corresponding time series object
        if not np.all(initial_firm_prices == initial_firm_prices[0]):
            raise ValueError("Initial prices must be equal.")
        ts = create_economy_timeseries(
            country_name=country_name,
            all_country_names=all_country_names,
            n_industries=n_industries,
            initial_firm_prices=initial_firm_prices[0],
            initial_firm_total_sales=initial_firm_total_sales,
            initial_firm_total_used_ii=initial_firm_total_used_ii,
            initial_total_taxes_on_products=initial_total_taxes_on_products,
            initial_change_in_firm_stock_inventories=initial_change_in_firm_stock_inventories,
            initial_total_operating_surplus_plus_wages=initial_total_operating_surplus_plus_wages,
            initial_individual_activity=initial_individual_activity,
            initial_cpi_inflation=initial_cpi_inflation,
            initial_ppi_inflation=initial_ppi_inflation,
            initial_nominal_house_price_index_growth=initial_nominal_house_price_index_growth,
            initial_real_rent_paid=initial_real_rent_paid,
            initial_imp_rent_paid=initial_imp_rent_paid,
            initial_hh_rental_income=initial_hh_rental_income,
            initial_hh_consumption=initial_hh_consumption,
            initial_gov_consumption=initial_gov_consumption,
            initial_cg_rent_received=initial_cg_rent_received,
            initial_cg_taxes_rental_income=initial_cg_taxes_rental_income,
            initial_sectoral_growth=initial_sectoral_growth,
            initial_sentiment=config["functions"]["sentiment"]["parameters"]["value"]["value"],
            initial_imports=initial_imports,
            initial_imports_by_country=initial_imports_by_country,
            initial_exports=initial_exports,
            initial_exports_by_country=initial_exports_by_country,
            export_taxes=export_taxes,
        )
        return cls(
            country_name,
            all_country_names,
            n_industries,
            functions,
            ts,
            states,
        )

    def set_estimates(
        self,
        exogenous_log_inflation: pd.DataFrame,
        exogenous_sectoral_growth: pd.DataFrame,
        exogenous_hpi_growth: pd.DataFrame,
    ) -> None:
        # Forecast CPI inflation
        historic_cpi_inflation = np.concatenate(
            (
                exogenous_log_inflation["Real CPI Inflation"].values,
                np.array(self.ts.historic("cpi_inflation")).flatten(),
            )
        )
        self.ts.estimated_cpi_inflation.append([self.functions["inflation"].forecast_inflation(historic_cpi_inflation)])

        # Forecast PPI inflation
        historic_ppi_inflation = np.concatenate(
            (
                exogenous_log_inflation["Real PPI Inflation"].values,
                np.array(self.ts.historic("ppi_inflation")).flatten(),
            )
        )
        self.ts.estimated_ppi_inflation.append([self.functions["inflation"].forecast_inflation(historic_ppi_inflation)])

        # Forecast industry-level growth
        estimated_growth = np.zeros(self.n_industries)
        for g in range(self.n_industries):
            if len(self.ts.historic("sectoral_growth")) == 0:
                historic_growth = exogenous_sectoral_growth[g].values
            else:
                historic_growth = np.concatenate(
                    (
                        exogenous_sectoral_growth[g].values,
                        np.array(self.ts.historic("sectoral_growth"))[:, g],
                    )
                )
            estimated_growth[g] = self.functions["growth"].forecast_growth(historic_growth)
        self.ts.estimated_sectoral_growth.append(estimated_growth)

        # Forecast house price index growth
        historic_hpi_growth = np.concatenate(
            (
                exogenous_hpi_growth["Nominal House Price Index Growth"].values,
                np.array(self.ts.historic("nominal_house_price_index_growth")).flatten(),
            )
        )
        self.ts.estimated_nominal_house_price_index_growth.append(
            [self.functions["house_price_index"].forecast_hpi_growth(historic_hpi_growth)]
        )

    def compute_sectoral_sentiment(self) -> None:
        self.ts.sectoral_sentiment.append(self.functions["sentiment"].compute_sentiment(n_industries=self.n_industries))

    @staticmethod
    def compute_number_of_employed_individuals(current_individual_activity_status: np.ndarray) -> int:
        return int(np.sum(current_individual_activity_status == ActivityStatus.EMPLOYED))

    def compute_price_indicators(
        self,
        firm_real_amount_bought: np.ndarray,
        firm_nominal_amount_spent: np.ndarray,
        household_real_amount_bought: np.ndarray,
        household_nominal_amount_spent: np.ndarray,
        government_real_amount_bought: np.ndarray,
        government_nominal_amount_spent: np.ndarray,
        firms_real_amount_bought_as_capital_goods: np.ndarray,
    ) -> None:
        # Current good prices
        current_goods_prices = np.zeros(self.n_industries)
        for g in range(self.n_industries):
            current_goods_prices[g] = self.compute_average_price(
                firm_real_amount_bought=firm_real_amount_bought,
                firm_nominal_amount_spent=firm_nominal_amount_spent,
                household_real_amount_bought=household_real_amount_bought,
                household_nominal_amount_spent=household_nominal_amount_spent,
                government_real_amount_bought=government_real_amount_bought,
                government_nominal_amount_spent=government_nominal_amount_spent,
                industry=g,
            )
        self.ts.good_prices.append(current_goods_prices)

        # PPI
        self.ts.ppi.append(
            [
                self.compute_average_price(
                    firm_real_amount_bought=firm_real_amount_bought,
                    firm_nominal_amount_spent=firm_nominal_amount_spent,
                    household_real_amount_bought=household_real_amount_bought,
                    household_nominal_amount_spent=household_nominal_amount_spent,
                    government_real_amount_bought=government_real_amount_bought,
                    government_nominal_amount_spent=government_nominal_amount_spent,
                    industry=None,
                )
            ]
        )

        # CPI
        consumption_by_industry_norm = household_nominal_amount_spent.sum(axis=0)
        if consumption_by_industry_norm.sum() == 0:
            self.ts.cpi.append(
                [
                    np.dot(
                        self.ts.current("good_prices"),
                        np.full(self.n_industries, 1.0 / self.n_industries),
                    )
                ]
            )
        else:
            consumption_by_industry_norm /= consumption_by_industry_norm.sum()
            self.ts.cpi.append([np.dot(self.ts.current("good_prices"), consumption_by_industry_norm)])

        # CFPI
        firm_inv_weights_norm = firms_real_amount_bought_as_capital_goods.sum(axis=0)
        if firm_inv_weights_norm.sum() == 0:
            self.ts.cfpi.append(
                [
                    np.dot(
                        self.ts.current("good_prices"),
                        np.full(self.n_industries, 1.0 / self.n_industries),
                    )
                ]
            )
        else:
            firm_inv_weights_norm /= firm_inv_weights_norm.sum()
            self.ts.cfpi.append([np.dot(self.ts.current("good_prices"), firm_inv_weights_norm)])

    def compute_inflation(self) -> None:
        # CPI inflation
        self.ts.cpi_inflation.append([np.log(self.ts.current("cpi")[0] / self.ts.prev("cpi")[0])])

        # PPI inflation
        self.ts.ppi_inflation.append([np.log(self.ts.current("ppi")[0] / self.ts.prev("ppi")[0])])

        # CFPI inflation
        self.ts.cfpi_inflation.append([np.log(self.ts.current("cfpi")[0] / self.ts.prev("cfpi")[0])])

        # Price inflation by industry
        inflation_by_industry = np.zeros(self.n_industries)
        for g in range(self.n_industries):
            inflation_by_industry[g] = np.log(self.ts.current("good_prices")[g] / self.ts.prev("good_prices")[g])
        self.ts.industry_inflation.append(inflation_by_industry)

    def compute_growth(
        self,
        current_production: np.ndarray,
        prev_production: np.ndarray,
        industries: np.ndarray,
    ) -> None:
        # Total growth
        self.ts.total_growth.append([(current_production.sum() - prev_production.sum()) / prev_production.sum()])

        # Growth by sector
        current_sectoral_growth = np.zeros(self.n_industries)
        for g in range(self.n_industries):
            current_total_output = current_production[industries == g].sum()
            prev_total_output = prev_production[industries == g].sum()
            if prev_total_output == 0:
                current_sectoral_growth[g] = 0.0
            else:
                current_sectoral_growth[g] = (current_total_output - prev_total_output) / prev_total_output
        self.ts.sectoral_growth.append(current_sectoral_growth)

    def compute_house_price_index(
        self,
        current_property_values: np.ndarray,
        previous_property_values: np.ndarray,
    ) -> None:
        self.ts.nominal_house_price_index_growth.append(
            [current_property_values.sum() / previous_property_values.sum() - 1.0]
        )

    def compute_average_price(
        self,
        firm_real_amount_bought: np.ndarray,
        firm_nominal_amount_spent: np.ndarray,
        household_real_amount_bought: np.ndarray,
        household_nominal_amount_spent: np.ndarray,
        government_real_amount_bought: np.ndarray,
        government_nominal_amount_spent: np.ndarray,
        industry: Optional[int],
    ) -> np.ndarray:
        if industry is None:
            if (
                firm_real_amount_bought.sum() + household_real_amount_bought.sum() + government_real_amount_bought.sum()
            ) == 0.0:
                return self.ts.current("ppi")[0]
            else:
                return (
                    firm_nominal_amount_spent.sum()
                    + household_nominal_amount_spent.sum()
                    + government_nominal_amount_spent.sum()
                ) / (
                    firm_real_amount_bought.sum()
                    + household_real_amount_bought.sum()
                    + government_real_amount_bought.sum()
                )
        else:
            if (
                firm_real_amount_bought[:, industry].sum()
                + household_real_amount_bought[:, industry].sum()
                + government_real_amount_bought[:, industry].sum()
                == 0.0
            ):
                return self.ts.current("good_prices")[industry]
            else:
                return (
                    firm_nominal_amount_spent[:, industry].sum()
                    + household_nominal_amount_spent[:, industry].sum()
                    + government_nominal_amount_spent[:, industry].sum()
                ) / (
                    firm_real_amount_bought[:, industry].sum()
                    + household_real_amount_bought[:, industry].sum()
                    + government_real_amount_bought[:, industry].sum()
                )

    def record_global_trade(
        self,
        firms: Firms,
        households: Households,
        government_entities: GovernmentEntities,
        tau_export: float,
    ) -> None:
        # Exports
        firm_industries = firms.states["Industry"]
        exports_before_taxes = np.zeros(self.n_industries)
        for rec_country in self.all_country_names:
            if rec_country == self.country_name:
                continue
            self.ts.dicts["exports_before_taxes_to_" + rec_country].append(
                np.array(
                    [
                        firms.ts.current("nominal_amount_sold_in_lcu_to_" + rec_country)[firm_industries == g].sum()
                        for g in range(self.n_industries)
                    ]
                )
            )
            exports_before_taxes += self.ts.current("exports_before_taxes_to_" + rec_country)
        self.ts.exports_before_taxes.append(exports_before_taxes)
        self.ts.exports.append((1 + tau_export) * self.ts.current("exports_before_taxes"))

        # Imports
        imports = np.zeros(self.n_industries)
        for sell_country in self.all_country_names:
            if sell_country == self.country_name:
                continue
            self.ts.dicts["imports_from_" + sell_country].append(
                firms.ts.current("nominal_amount_spent_in_lcu_to_" + sell_country).sum(axis=0)
                + households.ts.current("nominal_amount_spent_in_lcu_to_" + sell_country).sum(axis=0)
                + government_entities.ts.current("nominal_amount_spent_in_lcu_to_" + sell_country).sum(axis=0)
            )
            imports += self.ts.current("imports_from_" + sell_country)
        self.ts.imports.append(imports)

    def compute_labour_market_aggregates(
        self,
        current_individual_activity_status: np.ndarray,
        current_firm_labour_inputs: np.ndarray,
        current_desired_firm_labour_inputs: np.ndarray,
        num_ind_employed_before_cleaning: int,
        num_ind_newly_joining: int,
        num_ind_newly_leaving: int,
    ) -> None:
        # The unemployment rate
        self.ts.unemployment_rate.append(
            [
                np.sum(current_individual_activity_status == ActivityStatus.UNEMPLOYED)
                / (
                    np.sum(current_individual_activity_status == ActivityStatus.EMPLOYED)
                    + np.sum(current_individual_activity_status == ActivityStatus.UNEMPLOYED)
                )
            ]
        )

        # The participation rate
        self.ts.participation_rate.append(
            [
                (
                    np.sum(current_individual_activity_status == ActivityStatus.EMPLOYED)
                    + np.sum(current_individual_activity_status == ActivityStatus.UNEMPLOYED)
                )
                / len(current_individual_activity_status)
            ]
        )

        # The vacancy rate
        self.ts.vacancy_rate.append(
            [
                (current_desired_firm_labour_inputs.sum() - current_firm_labour_inputs.sum())
                / current_desired_firm_labour_inputs.sum()
            ]
        )

        # The job reallocation rate
        self.ts.job_reallocation_rate.append(
            [(num_ind_newly_joining + num_ind_newly_leaving) / num_ind_employed_before_cleaning]
        )

    def compute_rental_market_aggregates(
        self,
        real_rent_paid: np.ndarray,
        imp_rent_paid: np.ndarray,
        rental_income: np.ndarray,
    ) -> None:
        self.ts.total_real_rent_paid.append([real_rent_paid.sum()])
        self.ts.total_imp_rent_paid.append([imp_rent_paid.sum()])
        self.ts.total_real_rent_rec.append([rental_income.sum()])

    def compute_gdp(
        self,
        sales_minus_ii: float,
        taxes_on_products: float,
        rent_paid: float,
        rent_imputed: float,
        hh_consumption: float,
        gov_consumption: float,
        change_in_firm_stock_inventories: float,
        exports_minus_imports: float,
        operating_surplus_plus_wages: float,
        rent_received: float,
    ) -> None:
        self.ts.gdp_output.append([sales_minus_ii + taxes_on_products + rent_paid + rent_imputed])
        self.ts.gdp_expenditure.append(
            [
                change_in_firm_stock_inventories
                + hh_consumption
                + gov_consumption
                + exports_minus_imports
                + rent_paid
                + rent_imputed
            ]
        )
        self.ts.gdp_income.append([operating_surplus_plus_wages + taxes_on_products + rent_received + rent_imputed])
