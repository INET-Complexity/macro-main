"""Economy module for tracking and managing aggregate economic metrics.

This module implements the tracking and computation of key macroeconomic indicators
and aggregates across the entire economy. It serves as the central point for:

1. Price Level Tracking:
   - Consumer Price Index (CPI)
   - Producer Price Index (PPI)
   - Capital Formation Price Index (CFPI)
   - House Price Index (HPI)
   - Industry-specific price levels

2. Growth Metrics:
   - GDP components (output, expenditure, income approaches)
   - Sectoral growth rates
   - Total economic growth
   - Value added by industry

3. Labor Market Indicators:
   - Unemployment rate
   - Labor force participation
   - Job reallocation metrics
   - Vacancy rates

4. International Trade:
   - Import/export volumes
   - Trade balances by country
   - Exchange rate effects

5. Market Aggregates:
   - Housing market metrics
   - Credit market conditions
   - Goods market clearing

The module provides mechanisms for:
- Computing price indices and inflation rates
- Tracking GDP components and growth
- Managing international trade flows
- Recording labor market conditions
- Calculating market-specific aggregates

Implementation focuses on maintaining consistency across different
measurement approaches (output, expenditure, income) while handling
temporal evolution and cross-market interactions.
"""

from typing import Any, Optional

import h5py
import numpy as np
import pandas as pd

from macromodel.agents.central_government.central_government import CentralGovernment
from macromodel.agents.firms import Firms
from macromodel.agents.government_entities.government_entities import GovernmentEntities
from macromodel.agents.households.households import Households
from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.agents.individuals.individuals import Individuals
from macromodel.configurations import EconomyConfiguration
from macromodel.economy.economy_ts import create_economy_timeseries
from macromodel.exogenous.exogenous import Exogenous
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import functions_from_model, update_functions


class Economy:
    """Manages and tracks aggregate economic metrics and market-level interactions.

    This class serves as the central coordinator for tracking and computing
    macroeconomic indicators, market conditions, and aggregate metrics across
    the entire economy. It integrates data from all economic agents and markets
    to maintain consistent economic accounting.

    Key responsibilities include:
    1. Price Level Management:
       - Computing and tracking various price indices (CPI, PPI, CFPI)
       - Calculating inflation rates across sectors
       - Managing industry-specific price levels

    2. Economic Growth Tracking:
       - Computing GDP through multiple approaches
       - Tracking sectoral and total growth rates
       - Managing value-added calculations

    3. Labor Market Monitoring:
       - Tracking unemployment and participation rates
       - Computing job market dynamics
       - Recording vacancy statistics

    4. International Trade:
       - Managing import/export flows
       - Computing trade balances
       - Tracking exchange rate effects

    5. Market Integration:
       - Recording housing market metrics
       - Tracking credit conditions
       - Managing goods market clearing

    Attributes:
        country_name (str): Name identifier for the economy
        all_country_names (list[str]): List of all countries in the model
        n_industries (int): Number of industrial sectors
        functions (dict[str, Any]): Economic function implementations
        ts (TimeSeries): Time series data for economic metrics

    The class maintains consistency between different measurement approaches
    while handling temporal evolution and cross-market interactions. It ensures
    proper accounting of economic flows and stocks across all sectors.
    """

    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        functions: dict[str, Any],
        ts: TimeSeries,
    ):
        """Initialize an Economy instance.

        Args:
            country_name (str): Name identifier for the economy
            all_country_names (list[str]): List of all countries in the model
            n_industries (int): Number of industrial sectors
            functions (dict[str, Any]): Economic function implementations
            ts (TimeSeries): Time series data for economic metrics
        """
        self.country_name = country_name
        self.all_country_names = all_country_names
        self.n_industries = n_industries
        self.functions = functions
        self.n_industries = n_industries
        self.ts = ts

    @classmethod
    def from_agents(
        cls,
        country_name: str,
        all_country_names: list[str],
        economy_configuration: EconomyConfiguration,
        firms: Firms,
        households: Households,
        individuals: Individuals,
        government_entities: GovernmentEntities,
        central_government: CentralGovernment,
        exogenous: Exogenous,
        industry_vectors: pd.DataFrame,
    ):
        """Create an Economy instance from agent-level data and configurations.

        This factory method constructs an Economy by aggregating initial conditions
        from various economic agents and markets. It computes starting values for:
        - Price levels and production
        - Sales and intermediate inputs
        - Tax revenues and capital formation
        - Operating surplus and wages
        - Activity status and inflation rates
        - Housing market conditions
        - Trade flows and growth rates

        Args:
            country_name (str): Name identifier for the economy
            all_country_names (list[str]): List of all countries in the model
            economy_configuration (EconomyConfiguration): Model parameters
            firms (Firms): Collection of producing firms
            households (Households): Collection of household units
            individuals (Individuals): Population of individual agents
            government_entities (GovernmentEntities): Public sector bodies
            central_government (CentralGovernment): Fiscal authority
            exogenous (Exogenous): External economic conditions
            industry_vectors (pd.DataFrame): Industry-level initial conditions

        Returns:
            Economy: Newly constructed Economy instance with initialized metrics
        """
        initial_firm_prices = firms.ts.current("price")
        initial_total_output = (firms.ts.current("price") * firms.ts.current("production")).sum()
        initial_sectoral_firm_sales = np.bincount(
            firms.states["Industry"], weights=firms.ts.current("total_sales"), minlength=firms.n_industries
        )
        initial_sectoral_firm_used_ii = np.bincount(
            firms.states["Industry"],
            weights=firms.ts.current("used_intermediate_inputs_costs"),
            minlength=firms.n_industries,
        )
        initial_total_taxes_on_products = central_government.ts.current("taxes_on_products")[0]
        initial_total_taxes_on_production = central_government.ts.current("taxes_production")[0]
        initial_change_in_firm_stock_inventories = (
            firms.ts.current("total_inventory_change").sum()
            + firms.ts.current("total_intermediate_inputs_bought_costs").sum()
            - firms.ts.current("used_intermediate_inputs_costs").sum()
            # + firms.ts.current("total_capital_inputs_bought_costs").sum()  # TODO: NB -> Removed in Sam's code
        )
        initial_gross_fixed_capital_formation = (
            firms.ts.current("total_capital_inputs_bought_costs").sum()
            + (1 + central_government.states["Capital Formation Tax"]) * households.ts.current("investment").sum()
        )
        initial_total_operating_surplus = firms.ts.current("gross_operating_surplus_mixed_income").sum()
        initial_total_wages = firms.ts.current("total_wage").sum()

        initial_individual_activity = individuals.states["Activity Status"]
        initial_cpi_inflation = exogenous.ts.initial("cpi_inflation")[0]
        initial_ppi_inflation = exogenous.ts.initial("ppi_inflation")[0]
        initial_hpi_inflation = exogenous.ts.initial("hpi_inflation")[
            0
        ]  # Nominal House Price Index Growth is saved as hpi_inflation
        initial_real_rent_paid = households.ts.current("rent")
        initial_imp_rent_paid = households.ts.current("rent_imputed")
        initial_hh_rental_income = households.ts.current("income_rental")
        initial_hh_consumption = households.ts.current("total_consumption")[0]
        initial_gov_consumption = government_entities.ts.current("total_consumption")[0]
        initial_cg_rent_received = central_government.ts.current("total_rent_received")[0]
        initial_cg_taxes_rental_income = central_government.ts.current("taxes_rental_income")[0]
        # initial_sectoral_growth = exogenous.ts.initial("sectoral_growth")
        # initial_total_growth = exogenous.ts.initial("total_growth")[0]

        # TODO: again, this is hard-coded in Sam's code (default_initial_growth is a default value of _init_countries)
        #  We need to decide what to do with this, and where to put it.

        default_initial_growth: float = 0.01
        initial_total_growth = (
            (
                default_initial_growth
                if "Real Gross Output (Growth)" not in exogenous.national_accounts_during.columns
                else exogenous.national_accounts_during["Real Gross Output (Growth)"].values[0]
            ),
        )

        # initial_imports = exogenous.ts.initial("sectoral_imports")
        # initial_imports_by_country = {
        #     c: exogenous.ts.initial("sectoral_imports_from_" + c) for c in all_other_countries
        # }
        # initial_exports = exogenous.ts.initial("sectoral_exports")
        # initial_exports_by_country = {c: exogenous.ts.initial("sectoral_exports_to_" + c)
        # for c in all_other_countries}

        initial_imports = industry_vectors["Imports in LCU"].values.flatten()
        initial_imports_by_country = {
            c: industry_vectors["Imports in LCU from " + c].values.flatten()
            for c in all_country_names
            if c != country_name
        }
        initial_exports = industry_vectors["Exports in LCU"].values.flatten()
        initial_exports_by_country = {
            c: industry_vectors["Exports in LCU to " + c].values.flatten()
            for c in all_country_names
            if c != country_name
        }

        export_taxes = central_government.states["Export Tax"]

        initial_npl_ratio = 0.0

        ts = create_economy_timeseries(
            country_name=country_name,
            all_country_names=all_country_names,
            n_industries=firms.n_industries,
            initial_firm_prices=initial_firm_prices,  # .mean(),
            initial_firm_total_sales=initial_total_output,
            initial_sectoral_firm_sales=initial_sectoral_firm_sales,
            initial_sectoral_firm_used_ii=initial_sectoral_firm_used_ii,
            initial_total_taxes_on_products=initial_total_taxes_on_products,
            initial_total_taxes_on_production=initial_total_taxes_on_production,
            initial_change_in_firm_stock_inventories=initial_change_in_firm_stock_inventories,
            initial_gross_fixed_capital_formation=initial_gross_fixed_capital_formation,
            initial_total_operating_surplus=initial_total_operating_surplus,
            initial_total_wages=initial_total_wages,
            initial_individual_activity=initial_individual_activity,
            initial_cpi_inflation=initial_cpi_inflation,
            initial_ppi_inflation=initial_ppi_inflation,
            initial_hpi_inflation=initial_hpi_inflation,
            initial_real_rent_paid=initial_real_rent_paid,
            initial_imp_rent_paid=initial_imp_rent_paid,
            initial_hh_rental_income=initial_hh_rental_income,
            initial_hh_consumption=initial_hh_consumption,
            initial_gov_consumption=initial_gov_consumption,
            initial_cg_rent_received=initial_cg_rent_received,
            initial_cg_taxes_rental_income=initial_cg_taxes_rental_income,
            initial_imports=initial_imports,
            initial_imports_by_country=initial_imports_by_country,
            initial_exports=initial_exports,
            initial_exports_by_country=initial_exports_by_country,
            initial_total_growth=initial_total_growth[0],
            export_taxes=export_taxes,
            initial_npl_ratio=initial_npl_ratio,
        )

        functions = functions_from_model(economy_configuration.functions, loc="macromodel.economy")

        n_industries = firms.n_industries

        return cls(
            country_name,
            all_country_names,
            n_industries,
            functions,
            ts,
        )

    def reset(self, configuration: EconomyConfiguration) -> None:
        """Reset the economy's state and update function configurations.

        Resets all time series data and updates economic functions based on
        the provided configuration. This ensures a clean state for new simulations
        while maintaining consistent function implementations.

        Args:
            configuration (EconomyConfiguration): New model parameters
        """
        self.ts.reset()
        update_functions(
            model=configuration.functions,
            functions=self.functions,
            loc="macromodel.economy",
            force_reset=["growth", "house_price_index", "inflation"],
        )

    def set_estimates(
        self,
        exogenous_growth: np.ndarray,
        exogenous_inflation: pd.DataFrame,
        exogenous_hpi_growth: pd.DataFrame,
        forecasting_window: int,
        exogenous_cpi_inflation_during: np.ndarray,
        exogenous_ppi_inflation_during: np.ndarray,
        exogenous_growth_during: np.ndarray,
        default_growth: float = 0.005,
        default_inflation: float = 0.0,
        default_hpi_growth: float = 0.0,
        min_inflation: float = -0.1,
        max_inflation: float = 0.1,
        min_growth: float = -0.2,
        max_growth: float = 0.2,
        assume_zero_growth: bool = False,
        assume_zero_noise: bool = False,
    ) -> None:
        """Set economic forecasts and estimates for key indicators.

        Computes and sets forecasts for inflation (CPI, PPI), economic growth,
        and house price appreciation. Uses historical data and exogenous factors
        to generate estimates within specified bounds.

        The method handles:
        - CPI inflation forecasting
        - PPI inflation forecasting
        - Economic growth projection
        - House price index growth estimation

        Args:
            exogenous_growth (np.ndarray): External growth factors
            exogenous_inflation (pd.DataFrame): External inflation data
            exogenous_hpi_growth (pd.DataFrame): External house price growth
            forecasting_window (int): Number of periods to forecast
            exogenous_cpi_inflation_during (np.ndarray): CPI inflation factors
            exogenous_ppi_inflation_during (np.ndarray): PPI inflation factors
            exogenous_growth_during (np.ndarray): Growth factors during period
            default_growth (float, optional): Fallback growth rate. Defaults to 0.005.
            default_inflation (float, optional): Fallback inflation. Defaults to 0.0.
            default_hpi_growth (float, optional): Fallback HPI growth. Defaults to 0.0.
            min_inflation (float, optional): Lower inflation bound. Defaults to -0.1.
            max_inflation (float, optional): Upper inflation bound. Defaults to 0.1.
            min_growth (float, optional): Lower growth bound. Defaults to -0.2.
            max_growth (float, optional): Upper growth bound. Defaults to 0.2.
            assume_zero_growth (bool, optional): Force zero growth. Defaults to False.
            assume_zero_noise (bool, optional): Eliminate random variation. Defaults to False.
        """
        # Forecast CPI inflation
        historic_cpi_inflation = np.concatenate(
            (
                exogenous_inflation["CPI Inflation"].values[-forecasting_window:],
                np.array(self.ts.historic("cpi_inflation")).flatten(),
            )
        )
        if assume_zero_growth:
            self.ts.estimated_cpi_inflation.append([0.0])
        else:
            if len(historic_cpi_inflation[~np.isnan(historic_cpi_inflation)]) < 3:
                self.ts.estimated_cpi_inflation.append([default_inflation])
            else:
                estimated_cpi_inflation = (
                    np.exp(
                        self.functions["inflation"].forecast_inflation(
                            historic_inflation=historic_cpi_inflation,
                            exogenous_inflation=exogenous_cpi_inflation_during,
                            current_time=len(self.ts.historic("cpi")),
                            assume_zero_noise=assume_zero_noise,
                        )[0]
                    )
                    - 1.0
                )
                estimated_cpi_inflation = np.maximum(
                    min_inflation,
                    np.minimum(max_inflation, estimated_cpi_inflation),
                )
                assert not np.isnan(estimated_cpi_inflation)
                self.ts.estimated_cpi_inflation.append([estimated_cpi_inflation])

        # Forecast PPI inflation
        historic_ppi_inflation = np.concatenate(
            (
                exogenous_inflation["PPI Inflation"].values[-forecasting_window:],
                np.array(self.ts.historic("ppi_inflation")).flatten(),
            )
        )
        if assume_zero_growth:
            self.ts.estimated_ppi_inflation.append([0.0])
        else:
            if len(historic_ppi_inflation[~np.isnan(historic_ppi_inflation)]) < 3:
                self.ts.estimated_ppi_inflation.append([default_inflation])
            else:
                estimated_ppi_inflation = (
                    np.exp(
                        self.functions["inflation"].forecast_inflation(
                            historic_inflation=historic_ppi_inflation,
                            exogenous_inflation=exogenous_ppi_inflation_during,
                            current_time=len(self.ts.historic("ppi")),
                            assume_zero_noise=assume_zero_noise,
                        )[0]
                    )
                    - 1.0
                )
                estimated_ppi_inflation = np.maximum(
                    min_inflation,
                    np.minimum(max_inflation, estimated_ppi_inflation),
                )
                assert not np.isnan(estimated_ppi_inflation)
                self.ts.estimated_ppi_inflation.append([estimated_ppi_inflation])

        # Forecast growth
        if assume_zero_growth:
            self.ts.estimated_growth.append([0.0])
        else:
            if exogenous_growth is None:
                self.ts.estimated_growth.append([default_growth])
            else:
                historic_growth = np.concatenate(
                    (
                        exogenous_growth[-forecasting_window:],
                        np.array(self.ts.historic("total_growth")).flatten(),
                    )
                )
                estimated_growth = (
                    np.exp(
                        self.functions["growth"].forecast_growth(
                            historic_growth=historic_growth,
                            exogenous_growth=exogenous_growth_during,
                            current_time=len(self.ts.historic("ppi")),
                            assume_zero_noise=assume_zero_noise,
                        )[0]
                    )
                    - 1.0
                )
                estimated_growth = np.maximum(min_growth, np.minimum(max_growth, estimated_growth))
                assert not np.isnan(estimated_growth)
                self.ts.estimated_growth.append([estimated_growth])

        # Forecast house price index growth
        historic_hpi_growth = np.concatenate(
            (
                exogenous_hpi_growth["Nominal House Price Index Growth"].values[-forecasting_window:],
                np.array(self.ts.historic("hpi_inflation")).flatten(),
            )
        )
        if assume_zero_growth:
            self.ts.estimated_hpi_inflation.append([0.0])
        else:
            if len(historic_hpi_growth[~np.isnan(historic_hpi_growth)]) < 3:
                self.ts.estimated_hpi_inflation.append([default_hpi_growth])
            else:
                estimated_hpi_inflation = (
                    np.exp(
                        self.functions["house_price_index"].forecast_hpi_growth(
                            historic_hpi=historic_hpi_growth,
                            min_hpi_growth=min_growth,
                            max_hpi_growth=max_growth,
                            assume_zero_noise=assume_zero_noise,
                        )[0]
                    )
                    - 1.0
                )
                assert not np.isnan(estimated_hpi_inflation)
                self.ts.estimated_hpi_inflation.append([estimated_hpi_inflation])

    @staticmethod
    def compute_number_of_employed_individuals(
        current_individual_activity_status: np.ndarray,
    ) -> int:
        """Calculate the total number of employed individuals.

        Counts individuals with employment status in the current period,
        used for labor market statistics and economic indicators.

        Args:
            current_individual_activity_status (np.ndarray): Array of activity statuses

        Returns:
            int: Count of employed individuals
        """
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
        """Compute and update various price indices for the economy.

        Calculates and records:
        1. Industry-specific goods prices
        2. Producer Price Index (PPI)
        3. Consumer Price Index (CPI)
        4. Capital Formation Price Index (CFPI)

        The method uses real and nominal transaction data from firms,
        households, and government to compute weighted average prices
        and construct price indices relative to initial conditions.

        Args:
            firm_real_amount_bought (np.ndarray): Physical quantities bought by firms
            firm_nominal_amount_spent (np.ndarray): Money spent by firms
            household_real_amount_bought (np.ndarray): Physical quantities bought by households
            household_nominal_amount_spent (np.ndarray): Money spent by households
            government_real_amount_bought (np.ndarray): Physical quantities bought by government
            government_nominal_amount_spent (np.ndarray): Money spent by government
            firms_real_amount_bought_as_capital_goods (np.ndarray): Capital goods quantities
        """
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
                / self.ts.initial("initial_price")[0][0]
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
                    / self.ts.initial("initial_price")[0][0]
                ]
            )
        else:
            consumption_by_industry_norm /= consumption_by_industry_norm.sum()
            self.ts.cpi.append(
                [
                    np.dot(
                        self.ts.current("good_prices"),
                        consumption_by_industry_norm,
                    )
                    / self.ts.initial("initial_price")[0][0]
                ]
            )

        # CFPI
        firm_inv_weights_norm = firms_real_amount_bought_as_capital_goods.sum(axis=0)
        if firm_inv_weights_norm.sum() == 0:
            self.ts.cfpi.append(
                [
                    np.dot(
                        self.ts.current("good_prices"),
                        np.full(self.n_industries, 1.0 / self.n_industries),
                    )
                    / self.ts.initial("initial_price")[0][0]
                ]
            )
        else:
            firm_inv_weights_norm /= firm_inv_weights_norm.sum()
            self.ts.cfpi.append(
                [np.dot(self.ts.current("good_prices"), firm_inv_weights_norm) / self.ts.initial("initial_price")[0][0]]
            )

    def compute_inflation(self) -> None:
        """Calculate and record various inflation measures.

        Computes period-over-period inflation rates for:
        - Consumer Price Index (CPI)
        - Producer Price Index (PPI)
        - Capital Formation Price Index (CFPI)
        - Industry-specific price levels

        All rates are calculated as percentage changes from previous period.
        """
        # CPI inflation
        self.ts.cpi_inflation.append([self.ts.current("cpi")[0] / self.ts.prev("cpi")[0] - 1.0])

        # PPI inflation
        self.ts.ppi_inflation.append([self.ts.current("ppi")[0] / self.ts.prev("ppi")[0] - 1.0])

        # CFPI inflation
        self.ts.cfpi_inflation.append([self.ts.current("cfpi")[0] / self.ts.prev("cfpi")[0] - 1.0])

        # Price inflation by industry
        inflation_by_industry = np.zeros(self.n_industries)
        for g in range(self.n_industries):
            inflation_by_industry[g] = self.ts.current("good_prices")[g] / self.ts.prev("good_prices")[g] - 1.0
        self.ts.industry_inflation.append(inflation_by_industry)

    def compute_growth(
        self,
        current_production: np.ndarray,
        prev_production: np.ndarray,
        industries: np.ndarray,
    ) -> None:
        """Calculate and record economic growth rates.

        Computes both aggregate and sectoral growth rates based on
        production volumes. Handles special cases where previous
        production was zero.

        Args:
            current_production (np.ndarray): Current period production volumes
            prev_production (np.ndarray): Previous period production volumes
            industries (np.ndarray): Industry indices for sectoral mapping
        """
        # Total growth
        if prev_production.sum() == 0.0:
            self.ts.total_growth.append([0.0])
        else:
            self.ts.total_growth.append([(current_production.sum() - prev_production.sum()) / prev_production.sum()])
        if self.ts.current("total_growth")[0] < -1.0:
            print(current_production.sum(), prev_production.sum())
            print("--")
            print(self.ts.current("ppi_inflation"))
            print("---")

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
        """Calculate and record house price index and growth.

        Computes the change in aggregate property values and updates
        both the house price inflation rate and index level.

        Args:
            current_property_values (np.ndarray): Current property valuations
            previous_property_values (np.ndarray): Previous property valuations
        """
        if previous_property_values.sum() == 0:
            self.ts.hpi_inflation.append([0.0])
        else:
            self.ts.hpi_inflation.append([current_property_values.sum() / previous_property_values.sum() - 1.0])
        self.ts.hpi.append([(1 + self.ts.current("hpi_inflation")[0]) * self.ts.current("hpi")[0]])

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
        """Calculate weighted average price across all buyers or for a specific industry.

        Computes the average price as the ratio of total nominal spending to
        total real quantities purchased. Handles both economy-wide averages
        and industry-specific calculations.

        If no transactions occurred or if total quantities are zero, returns
        the current price (PPI for economy-wide, industry price for specific sector).

        Args:
            firm_real_amount_bought (np.ndarray): Physical quantities bought by firms
            firm_nominal_amount_spent (np.ndarray): Money spent by firms
            household_real_amount_bought (np.ndarray): Physical quantities bought by households
            household_nominal_amount_spent (np.ndarray): Money spent by households
            government_real_amount_bought (np.ndarray): Physical quantities bought by government
            government_nominal_amount_spent (np.ndarray): Money spent by government
            industry (Optional[int]): Industry index, or None for economy-wide average

        Returns:
            np.ndarray: Computed average price(s)
        """
        if industry is None:
            if (
                firm_real_amount_bought.sum() + household_real_amount_bought.sum() + government_real_amount_bought.sum()
            ) == 0.0 or (
                firm_nominal_amount_spent.sum()
                + household_nominal_amount_spent.sum()
                + government_nominal_amount_spent.sum()
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
            ) or (
                firm_nominal_amount_spent[:, industry].sum()
                + household_nominal_amount_spent[:, industry].sum()
                + government_nominal_amount_spent[:, industry].sum()
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
        """Record international trade flows and balances.

        Tracks bilateral trade flows between countries, computing:
        - Exports before and after taxes by destination
        - Imports by source country and sector
        - Total trade volumes and balances

        Args:
            firms (Firms): Collection of producing firms
            households (Households): Collection of household units
            government_entities (GovernmentEntities): Public sector bodies
            tau_export (float): Export tax rate
        """
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
        """Calculate and record key labor market indicators.

        Computes and updates:
        1. Unemployment rate and its growth
        2. Labor force participation rate and its growth
        3. Job vacancy rate and its growth
        4. Job reallocation rate and its growth

        The method tracks both levels and dynamics of labor market
        conditions, handling special cases where denominators may be zero.

        Args:
            current_individual_activity_status (np.ndarray): Current activity status
            current_firm_labour_inputs (np.ndarray): Actual labor employed
            current_desired_firm_labour_inputs (np.ndarray): Desired labor demand
            num_ind_employed_before_cleaning (int): Prior employment count
            num_ind_newly_joining (int): New hires count
            num_ind_newly_leaving (int): Separations count
        """
        # The unemployment rate
        self.ts.unemployment_rate.append(
            [
                np.sum(current_individual_activity_status == ActivityStatus.UNEMPLOYED)
                / (
                    np.sum(current_individual_activity_status == ActivityStatus.EMPLOYED)
                    + np.sum(current_individual_activity_status == ActivityStatus.UNEMPLOYED)
                    + np.sum(current_individual_activity_status == ActivityStatus.FIRM_INVESTOR)
                    + np.sum(current_individual_activity_status == ActivityStatus.BANK_INVESTOR)
                )
            ]
        )
        if self.ts.prev("unemployment_rate")[0] == 0.0:
            self.ts.unemployment_rate_growth.append([0.0])
        else:
            self.ts.unemployment_rate_growth.append(
                [self.ts.current("unemployment_rate")[0] / self.ts.prev("unemployment_rate")[0] - 1.0]
            )

        # The participation rate
        self.ts.participation_rate.append(
            [
                (
                    np.sum(current_individual_activity_status == ActivityStatus.EMPLOYED)
                    + np.sum(current_individual_activity_status == ActivityStatus.UNEMPLOYED)
                    + np.sum(current_individual_activity_status == ActivityStatus.FIRM_INVESTOR)
                    + np.sum(current_individual_activity_status == ActivityStatus.BANK_INVESTOR)
                )
                / len(current_individual_activity_status)
            ]
        )
        if self.ts.prev("participation_rate")[0] == 0.0:
            self.ts.participation_rate_growth.append([0.0])
        else:
            self.ts.participation_rate_growth.append(
                [self.ts.current("participation_rate")[0] / self.ts.prev("participation_rate")[0] - 1.0]
            )

        # The vacancy rate
        if current_desired_firm_labour_inputs.sum() == 0.0:
            self.ts.vacancy_rate.append([0.0])
        else:
            self.ts.vacancy_rate.append(
                [
                    (current_desired_firm_labour_inputs.sum() - current_firm_labour_inputs.sum())
                    / current_desired_firm_labour_inputs.sum()
                ]
            )
        if self.ts.prev("vacancy_rate")[0] == 0.0:
            self.ts.vacancy_rate_growth.append([0.0])
        else:
            self.ts.vacancy_rate_growth.append(
                [self.ts.current("vacancy_rate")[0] / self.ts.prev("vacancy_rate")[0] - 1.0]
            )

        # The job reallocation rate
        if num_ind_employed_before_cleaning == 0.0:
            self.ts.job_reallocation_rate.append([0.0])
        else:
            self.ts.job_reallocation_rate.append(
                [(num_ind_newly_joining + num_ind_newly_leaving) / num_ind_employed_before_cleaning]
            )
        if self.ts.prev("job_reallocation_rate")[0] == 0.0:
            self.ts.job_reallocation_rate_growth.append([0.0])
        else:
            self.ts.job_reallocation_rate_growth.append(
                [self.ts.current("job_reallocation_rate")[0] / self.ts.prev("job_reallocation_rate")[0] - 1.0]
            )

    def compute_rental_market_aggregates(
        self,
        real_rent_paid: np.ndarray,
        imp_rent_paid: np.ndarray,
        rental_income: np.ndarray,
    ) -> None:
        """Calculate and record rental market totals.

        Updates aggregate measures for:
        - Total real rent payments
        - Total imputed rent
        - Total rental income received

        Args:
            real_rent_paid (np.ndarray): Actual rent payments
            imp_rent_paid (np.ndarray): Imputed rent values
            rental_income (np.ndarray): Income from rental properties
        """
        self.ts.total_real_rent_paid.append([real_rent_paid.sum()])
        self.ts.total_imp_rent_paid.append([imp_rent_paid.sum()])
        self.ts.total_real_rent_rec.append([rental_income.sum()])

    def compute_gdp(
        self,
        total_output: float,
        sectoral_sales: np.ndarray,
        sectoral_intermediate_consumption: np.ndarray,
        taxes_on_products: float,
        taxes_on_production: float,
        rent_paid: float,
        rent_imputed: float,
        hh_consumption: float,
        gov_consumption: float,
        change_in_inventories: float,
        gross_fixed_capital_formation: float,
        exports: float,
        imports: float,
        operating_surplus: float,
        wages: float,
        rent_received: float,
        running_multiple_countries: bool,
        always_adjust: bool = True,
    ) -> None:
        """Calculate GDP through multiple approaches and record components.

        Computes GDP using three approaches:
        1. Output approach: Production less intermediate consumption
        2. Expenditure approach: Final spending components
        3. Income approach: Factor payments and operating surplus

        Also calculates and records:
        - Sectoral value added and growth rates
        - GDP components and their growth rates
        - Trade balance adjustments
        - Consistency checks between approaches

        The method maintains National Accounts identities and handles
        adjustments for multi-country scenarios.

        Args:
            total_output (float): Total production value
            sectoral_sales (np.ndarray): Sales by industry
            sectoral_intermediate_consumption (np.ndarray): Intermediate inputs
            taxes_on_products (float): Product taxes net of subsidies
            taxes_on_production (float): Production taxes
            rent_paid (float): Actual rent payments
            rent_imputed (float): Imputed rent values
            hh_consumption (float): Household consumption
            gov_consumption (float): Government consumption
            change_in_inventories (float): Inventory changes
            gross_fixed_capital_formation (float): Fixed investment
            exports (float): Export value
            imports (float): Import value
            operating_surplus (float): Operating surplus and mixed income
            wages (float): Compensation of employees
            rent_received (float): Rental income received
            running_multiple_countries (bool): Multi-country simulation flag
            always_adjust (bool, optional): Force trade adjustments. Defaults to True.
        """
        self.ts.gdp_output.append(
            [
                total_output
                - sectoral_intermediate_consumption.sum()
                - taxes_on_production
                + taxes_on_products
                + rent_paid
                + rent_imputed
            ]
        )
        if self.ts.prev("gdp_output")[0] == 0.0:
            self.ts.gdp_output_growth.append([0.0])
        else:
            self.ts.gdp_output_growth.append([self.ts.current("gdp_output")[0] / self.ts.prev("gdp_output")[0] - 1.0])
        self.ts.total_output.append([total_output])
        if self.ts.prev("total_output")[0] == 0.0:
            self.ts.total_output_growth.append([0.0])
        else:
            self.ts.total_output_growth.append(
                [self.ts.current("total_output")[0] / self.ts.prev("total_output")[0] - 1.0]
            )
        self.ts.total_intermediate_consumption.append([sectoral_intermediate_consumption.sum()])
        if self.ts.prev("total_intermediate_consumption")[0] == 0.0:
            self.ts.total_intermediate_consumption_growth.append([0.0])
        else:
            self.ts.total_intermediate_consumption_growth.append(
                [
                    self.ts.current("total_intermediate_consumption")[0]
                    / self.ts.prev("total_intermediate_consumption")[0]
                    - 1.0
                ]
            )
        self.ts.total_gross_value_added.append(
            [total_output - sectoral_intermediate_consumption.sum() - taxes_on_production]
        )
        if self.ts.prev("total_gross_value_added")[0] == 0.0:
            self.ts.total_gross_value_added_growth.append([0.0])
        else:
            self.ts.total_gross_value_added_growth.append(
                [self.ts.current("total_gross_value_added")[0] / self.ts.prev("total_gross_value_added")[0] - 1.0]
            )
        self.ts.total_gross_value_added_a.append([sectoral_sales[0] - sectoral_intermediate_consumption[0]])
        if self.ts.prev("total_gross_value_added_a")[0] == 0.0:
            self.ts.total_gross_value_added_a_growth.append([0.0])
        else:
            self.ts.total_gross_value_added_a_growth.append(
                [self.ts.current("total_gross_value_added_a")[0] / self.ts.prev("total_gross_value_added_a")[0] - 1.0]
            )
        self.ts.total_gross_value_added_bcde.append(
            [sectoral_sales[1:5].sum() - sectoral_intermediate_consumption[1:5].sum()]
        )
        if self.ts.prev("total_gross_value_added_bcde")[0] == 0.0:
            self.ts.total_gross_value_added_bcde_growth.append([0.0])
        else:
            self.ts.total_gross_value_added_bcde_growth.append(
                [
                    self.ts.current("total_gross_value_added_bcde")[0] / self.ts.prev("total_gross_value_added_bcde")[0]
                    - 1.0
                ]
            )
        self.ts.total_gross_value_added_c.append([sectoral_sales[2] - sectoral_intermediate_consumption[2]])
        if self.ts.prev("total_gross_value_added_c")[0] == 0.0:
            self.ts.total_gross_value_added_c_growth.append([0.0])
        else:
            self.ts.total_gross_value_added_c_growth.append(
                [self.ts.current("total_gross_value_added_c")[0] / self.ts.prev("total_gross_value_added_c")[0] - 1.0]
            )
        self.ts.total_gross_value_added_f.append([sectoral_sales[5] - sectoral_intermediate_consumption[5]])
        if self.ts.prev("total_gross_value_added_f")[0] == 0.0:
            self.ts.total_gross_value_added_f_growth.append([0.0])
        else:
            self.ts.total_gross_value_added_f_growth.append(
                [self.ts.current("total_gross_value_added_f")[0] / self.ts.prev("total_gross_value_added_f")[0] - 1.0]
            )
        self.ts.total_gross_value_added_ghijklmnopqrstu.append(
            [sectoral_sales[6:].sum() - sectoral_intermediate_consumption[6:].sum()]
        )
        if self.ts.prev("total_gross_value_added_ghijklmnopqrstu")[0] == 0.0:
            self.ts.total_gross_value_added_ghijklmnopqrstu_growth.append([0.0])
        else:
            self.ts.total_gross_value_added_ghijklmnopqrstu_growth.append(
                [
                    self.ts.current("total_gross_value_added_ghijklmnopqrstu")[0]
                    / self.ts.prev("total_gross_value_added_ghijklmnopqrstu")[0]
                    - 1.0
                ]
            )
        self.ts.total_gross_value_added_ghi.append(
            [sectoral_sales[6:9].sum() - sectoral_intermediate_consumption[6:9].sum()]
        )
        if self.ts.prev("total_gross_value_added_ghi")[0] == 0.0:
            self.ts.total_gross_value_added_ghi_growth.append([0.0])
        else:
            self.ts.total_gross_value_added_ghi_growth.append(
                [
                    self.ts.current("total_gross_value_added_ghi")[0] / self.ts.prev("total_gross_value_added_ghi")[0]
                    - 1.0
                ]
            )
        self.ts.total_gross_value_added_j.append([sectoral_sales[9] - sectoral_intermediate_consumption[9]])
        if self.ts.prev("total_gross_value_added_j")[0] == 0.0:
            self.ts.total_gross_value_added_j_growth.append([0.0])
        else:
            self.ts.total_gross_value_added_j_growth.append(
                [self.ts.current("total_gross_value_added_j")[0] / self.ts.prev("total_gross_value_added_j")[0] - 1.0]
            )
        self.ts.total_gross_value_added_k.append([sectoral_sales[10] - sectoral_intermediate_consumption[10]])
        if self.ts.prev("total_gross_value_added_k")[0] == 0.0:
            self.ts.total_gross_value_added_k_growth.append([0.0])
        else:
            self.ts.total_gross_value_added_k_growth.append(
                [self.ts.current("total_gross_value_added_k")[0] / self.ts.prev("total_gross_value_added_k")[0] - 1.0]
            )
        self.ts.total_gross_value_added_l.append([sectoral_sales[11] - sectoral_intermediate_consumption[11]])
        if self.ts.prev("total_gross_value_added_l")[0] == 0.0:
            self.ts.total_gross_value_added_l_growth.append([0.0])
        else:
            self.ts.total_gross_value_added_l_growth.append(
                [self.ts.current("total_gross_value_added_l")[0] / self.ts.prev("total_gross_value_added_l")[0] - 1.0]
            )
        self.ts.total_gross_value_added_mn.append(
            [sectoral_sales[12:14].sum() - sectoral_intermediate_consumption[12:14].sum()]
        )
        if self.ts.prev("total_gross_value_added_mn")[0] == 0.0:
            self.ts.total_gross_value_added_mn_growth.append([0.0])
        else:
            self.ts.total_gross_value_added_mn_growth.append(
                [self.ts.current("total_gross_value_added_mn")[0] / self.ts.prev("total_gross_value_added_mn")[0] - 1.0]
            )
        self.ts.total_gross_value_added_opq.append(
            [sectoral_sales[14:17].sum() - sectoral_intermediate_consumption[14:17].sum()]
        )
        if self.ts.prev("total_gross_value_added_opq")[0] == 0.0:
            self.ts.total_gross_value_added_opq_growth.append([0.0])
        else:
            self.ts.total_gross_value_added_opq_growth.append(
                [
                    self.ts.current("total_gross_value_added_opq")[0] / self.ts.prev("total_gross_value_added_opq")[0]
                    - 1.0
                ]
            )
        self.ts.total_gross_value_added_rstu.append(
            [sectoral_sales[17:].sum() - sectoral_intermediate_consumption[17:].sum()]
        )
        if self.ts.prev("total_gross_value_added_rstu")[0] == 0.0:
            self.ts.total_gross_value_added_rstu_growth.append([0.0])
        else:
            self.ts.total_gross_value_added_rstu_growth.append(
                [
                    self.ts.current("total_gross_value_added_rstu")[0] / self.ts.prev("total_gross_value_added_rstu")[0]
                    - 1.0
                ]
            )
        self.ts.total_taxes_less_subsidies_on_products.append([taxes_on_products])
        if self.ts.prev("total_taxes_less_subsidies_on_products")[0] == 0.0:
            self.ts.total_taxes_less_subsidies_on_products_growth.append([0.0])
        else:
            self.ts.total_taxes_less_subsidies_on_products_growth.append(
                [
                    self.ts.current("total_taxes_less_subsidies_on_products")[0]
                    / self.ts.prev("total_taxes_less_subsidies_on_products")[0]
                    - 1.0
                ]
            )
        self.ts.total_taxes_on_production.append([taxes_on_production])
        if self.ts.prev("total_taxes_on_production")[0] == 0.0:
            self.ts.total_taxes_on_production_growth.append([0.0])
        else:
            self.ts.total_taxes_on_production_growth.append(
                [self.ts.current("total_taxes_on_production")[0] / self.ts.prev("total_taxes_on_production")[0] - 1.0]
            )
        gdp_expenditure = (
            change_in_inventories
            + gross_fixed_capital_formation
            + hh_consumption
            + gov_consumption
            + exports
            - imports
        )
        self.ts.total_household_fce.append([hh_consumption])
        if self.ts.prev("total_household_fce")[0] == 0.0:
            self.ts.total_household_fce_growth.append([0.0])
        else:
            self.ts.total_household_fce_growth.append(
                [self.ts.current("total_household_fce")[0] / self.ts.prev("total_household_fce")[0] - 1.0]
            )
        self.ts.total_government_fce.append([gov_consumption])
        if self.ts.prev("total_government_fce")[0] == 0.0:
            self.ts.total_government_fce_growth.append([0.0])
        else:
            self.ts.total_government_fce_growth.append(
                [self.ts.current("total_government_fce")[0] / self.ts.prev("total_government_fce")[0] - 1.0]
            )
        self.ts.total_gross_fixed_capital_formation.append([gross_fixed_capital_formation])
        if self.ts.prev("total_gross_fixed_capital_formation")[0] == 0.0:
            self.ts.total_gross_fixed_capital_formation_growth.append([0.0])
        else:
            self.ts.total_gross_fixed_capital_formation_growth.append(
                [
                    self.ts.current("total_gross_fixed_capital_formation")[0]
                    / self.ts.prev("total_gross_fixed_capital_formation")[0]
                    - 1.0
                ]
            )
        self.ts.total_changes_in_inventories.append([change_in_inventories])
        if self.ts.prev("total_changes_in_inventories")[0] == 0:
            self.ts.total_changes_in_inventories_growth.append([0.0])
        else:
            self.ts.total_changes_in_inventories_growth.append(
                [
                    self.ts.current("total_changes_in_inventories")[0] / self.ts.prev("total_changes_in_inventories")[0]
                    - 1.0
                ]
            )
        self.ts.gdp_income.append(
            [
                operating_surplus
                + wages
                + taxes_on_products
                + rent_received
                + rent_imputed
            ]
        )
        if self.ts.prev("gdp_income")[0] == 0.0:
            self.ts.gdp_income_growth.append([0.0])
        else:
            self.ts.gdp_income_growth.append([self.ts.current("gdp_income")[0] / self.ts.prev("gdp_income")[0] - 1.0])
        self.ts.total_gross_operating_surplus_and_mixed_income.append([operating_surplus])
        if self.ts.prev("total_gross_operating_surplus_and_mixed_income")[0] == 0.0:
            self.ts.total_gross_operating_surplus_and_mixed_income_growth.append([0.0])
        else:
            self.ts.total_gross_operating_surplus_and_mixed_income_growth.append(
                [
                    self.ts.current("total_gross_operating_surplus_and_mixed_income")[0]
                    / self.ts.prev("total_gross_operating_surplus_and_mixed_income")[0]
                    - 1.0
                ]
            )
        self.ts.total_compensation_of_employees.append([wages])
        if self.ts.prev("total_compensation_of_employees")[0] == 0.0:
            self.ts.total_compensation_of_employees_growth.append([0.0])
        else:
            self.ts.total_compensation_of_employees_growth.append(
                [
                    self.ts.current("total_compensation_of_employees")[0]
                    / self.ts.prev("total_compensation_of_employees")[0]
                    - 1.0
                ]
            )

        # Some adjustments may be necessary
        if running_multiple_countries or always_adjust:
            if gdp_expenditure > self.ts.current("gdp_output")[0]:
                imports += gdp_expenditure - self.ts.current("gdp_output")[0]
            else:
                exports += self.ts.current("gdp_output")[0] - gdp_expenditure

        # Update exports, imports, and expenditure
        self.ts.total_exports.append([exports])
        if self.ts.prev("total_exports")[0] == 0:
            self.ts.total_exports_growth.append([0.0])
        else:
            self.ts.total_exports_growth.append(
                [self.ts.current("total_exports")[0] / self.ts.prev("total_exports")[0] - 1.0]
            )
        self.ts.total_imports.append([imports])
        if self.ts.prev("total_imports")[0] == 0.0:
            self.ts.total_imports_growth.append([0.0])
        else:
            self.ts.total_imports_growth.append(
                [self.ts.current("total_imports")[0] / self.ts.prev("total_imports")[0] - 1.0]
            )
        self.ts.gdp_expenditure.append(
            [
                change_in_inventories
                + gross_fixed_capital_formation
                + hh_consumption
                + gov_consumption
                + exports
                - imports
                # + rent_paid
                # + rent_imputed
            ]
        )
        if self.ts.prev("gdp_expenditure")[0] == 0.0:
            self.ts.gdp_expenditure_growth.append([0.0])
        else:
            self.ts.gdp_expenditure_growth.append(
                [self.ts.current("gdp_expenditure")[0] / self.ts.prev("gdp_expenditure")[0] - 1.0]
            )

        # GDP sanity check
        if self.ts.current("gdp_output")[0] > 1e6:
            assert np.isclose(
                self.ts.current("gdp_output")[0],
                self.ts.current("gdp_expenditure")[0],
            )

    def save_to_h5(self, group: h5py.Group):
        """Save economy time series data to HDF5 format.

        Args:
            group (h5py.Group): HDF5 group to save data into
        """
        self.ts.write_to_h5("economy", group)

    def total_imports(self):
        """Get aggregate imports time series.

        Returns:
            float: Total imports value
        """
        return self.ts.get_aggregate("imports")

    def total_exports(self):
        """Get aggregate exports time series.

        Returns:
            float: Total exports value
        """
        return self.ts.get_aggregate("exports")

    def total_cpi_inflation(self):
        """Get aggregate CPI inflation time series.

        Returns:
            float: CPI inflation rate
        """
        return self.ts.get_aggregate("cpi")

    def total_ppi_inflation(self):
        """Get aggregate PPI inflation time series.

        Returns:
            float: PPI inflation rate
        """
        return self.ts.get_aggregate("ppi")

    def total_cfpi_inflation(self):
        """Get aggregate CFPI inflation time series.

        Returns:
            float: CFPI inflation rate
        """
        return self.ts.get_aggregate("cfpi")

    def unemployment_rate(self):
        """Get unemployment rate time series.

        Returns:
            float: Unemployment rate
        """
        return self.ts.get_aggregate("unemployment_rate")

    def gdp_expenditure(self):
        """Get GDP expenditure approach time series.

        Returns:
            float: GDP value from expenditure approach
        """
        return self.ts.get_aggregate("gdp_expenditure")

    def gdp_output(self):
        """Get GDP output approach time series.

        Returns:
            float: GDP value from output approach
        """
        return self.ts.get_aggregate("gdp_output")
