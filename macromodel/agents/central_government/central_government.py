"""Central Government agent implementation for macroeconomic modeling.

This module implements the central government agent, which manages:
- Tax collection and administration
- Social benefits distribution
- Fiscal policy implementation
- Government debt management

The central government plays a crucial role in:
- Revenue generation through various tax instruments
- Social welfare through benefits and transfers
- Economic stabilization through fiscal policy
- Public finance management
"""

from typing import Any

import h5py
import numpy as np

from macro_data import SyntheticCentralGovernment
from macro_data.processing import TaxData
from macromodel.agents.agent import Agent
from macromodel.agents.central_government.central_government_ts import (
    create_central_government_timeseries,
)
from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.configurations import CentralGovernmentConfiguration
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import functions_from_model, update_functions


class CentralGovernment(Agent):
    """Central Government agent responsible for fiscal policy and social benefits.

    This class implements government fiscal operations including:
    - Tax collection (VAT, income, corporate, etc.)
    - Social benefit distribution (unemployment, other transfers)
    - Public finance management (revenue, deficit, debt)

    The agent manages multiple tax instruments:
    - Value-added Tax (VAT)
    - Income Tax
    - Corporate Tax
    - Social Insurance Contributions
    - Export and Capital Formation Taxes

    Attributes:
        functions (dict[str, Any]): Mapping of function names to implementations
        states (dict[str, float | np.ndarray]): Current state variables including
            tax rates and benefit models
        ts (TimeSeries): Time series data for government variables
    """

    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        functions: dict[str, Any],
        ts: TimeSeries,
        states: dict[str, float | np.ndarray | list[np.ndarray]],
    ):
        """Initialize the Central Government agent.

        Args:
            country_name (str): Name of the country this government represents
            all_country_names (list[str]): List of all countries in the model
            n_industries (int): Number of industries in the economy
            functions (dict[str, Any]): Function implementations for government operations
            ts (TimeSeries): Time series data for tracking variables
            states (dict[str, float | np.ndarray]): State variables including tax rates
        """
        super().__init__(
            country_name,
            all_country_names,
            n_industries,
            0,
            0,
            ts,
            states,
        )
        self.functions = functions

    @classmethod
    def from_pickled_agent(
        cls,
        synthetic_central_government: SyntheticCentralGovernment,
        configuration: CentralGovernmentConfiguration,
        n_industries: int,
        country_name: str,
        all_country_names: list[str],
        tax_data: TaxData,
        number_of_unemployed_individuals: int,
        taxes_net_subsidies: np.ndarray,
    ):
        """Create a Central Government instance from pickled data.

        Initializes the government with:
        - Tax rates from historical data
        - Benefit models from synthetic data
        - Configuration parameters
        - Country-specific settings

        Args:
            synthetic_central_government (SyntheticCentralGovernment): Synthetic data
            configuration (CentralGovernmentConfiguration): Configuration parameters
            n_industries (int): Number of industries
            country_name (str): Country name
            all_country_names (list[str]): All country names
            tax_data (TaxData): Historical tax rate data
            number_of_unemployed_individuals (int): Count of unemployed
            taxes_net_subsidies (np.ndarray): Net tax rates by sector

        Returns:
            CentralGovernment: Initialized government agent
        """
        functions = functions_from_model(model=configuration.functions, loc="macromodel.agents.central_government")

        states = {
            "Value-added Tax": tax_data.value_added_tax,
            "Export Tax": tax_data.export_tax,
            "Employer Social Insurance Tax": tax_data.employer_social_insurance_tax,
            "Employee Social Insurance Tax": tax_data.employee_social_insurance_tax,
            "Profit Tax": tax_data.profit_tax,
            "Income Tax": tax_data.income_tax,
            "Capital Formation Tax": tax_data.capital_formation_tax,
            "Taxes Less Subsidies Rates": taxes_net_subsidies,
            "unemployment_benefits_model": synthetic_central_government.unemployment_benefits_model,
            "other_benefits_model": synthetic_central_government.other_benefits_model,
        }

        data = (synthetic_central_government.central_gov_data.astype(float)).rename_axis("Central Government ID")

        ts = create_central_government_timeseries(
            data=data,
            number_of_unemployed_individuals=number_of_unemployed_individuals,
        )

        return cls(
            country_name,
            all_country_names,
            n_industries,
            functions,
            ts,
            states,
        )

    def reset(self, configuration: CentralGovernmentConfiguration):
        """Reset the government agent to initial state.

        Resets all state variables and updates function implementations
        based on the provided configuration.

        Args:
            configuration (CentralGovernmentConfiguration): New configuration
                parameters for the reset state
        """
        self.gen_reset()
        update_functions(
            model=configuration.functions, loc="macromodel.agents.central_government", functions=self.functions
        )

    def update_benefits(
        self,
        historic_ppi_inflation: list[np.ndarray],
        exogenous_ppi_inflation: np.ndarray,
        current_estimated_ppi_inflation: float,
        current_unemployment_rate: float,
        current_estimated_growth: float,
    ) -> None:
        """Update social benefit levels based on economic conditions.

        Adjusts both unemployment benefits and other social transfers
        considering:
        - Historical and expected inflation
        - Current unemployment rate
        - Economic growth estimates

        Args:
            historic_ppi_inflation (list[np.ndarray]): Past inflation rates
            exogenous_ppi_inflation (np.ndarray): External inflation factors
            current_estimated_ppi_inflation (float): Current inflation estimate
            current_unemployment_rate (float): Current unemployment rate
            current_estimated_growth (float): Estimated economic growth
        """
        all_ppi_inflation = np.concatenate(
            (
                exogenous_ppi_inflation,
                np.array(historic_ppi_inflation).flatten(),
                [current_estimated_ppi_inflation],
            )
        )

        # Unemployment benefits
        self.ts.unemployment_benefits_by_individual.append(
            [
                self.functions["social_benefits"].compute_unemployment_benefits(
                    prev_unemployment_benefits=self.ts.current("unemployment_benefits_by_individual")[0],
                    historic_ppi_inflation=all_ppi_inflation,
                    current_estimated_growth=current_estimated_growth,
                    current_unemployment_rate=current_unemployment_rate,
                    model=self.states["unemployment_benefits_model"],
                )
            ]
        )

        # Regular social transfers to households
        self.ts.total_other_benefits.append(
            [
                self.functions["social_benefits"].compute_regular_transfer_to_households(
                    prev_regular_transfer_to_households=self.ts.current("total_other_benefits")[0],
                    historic_ppi_inflation=all_ppi_inflation,
                    current_estimated_growth=current_estimated_growth,
                    current_unemployment_rate=current_unemployment_rate,
                    model=self.states["other_benefits_model"],
                )
            ]
        )

    def distribute_unemployment_benefits_to_individuals(
        self,
        current_individual_activity_status: np.ndarray,
    ) -> np.ndarray:
        """Distribute unemployment benefits to eligible individuals.

        Allocates unemployment benefits to individuals based on their
        current activity status (employed vs. unemployed).

        Args:
            current_individual_activity_status (np.ndarray): Activity status
                for each individual

        Returns:
            np.ndarray: Unemployment benefits by individual (zero for employed)
        """
        unemployment_benefits = np.zeros(current_individual_activity_status.shape)
        unemployment_benefits[current_individual_activity_status == ActivityStatus.UNEMPLOYED] = self.ts.current(
            "unemployment_benefits_by_individual"
        )[0]
        return unemployment_benefits.astype(float)

    def compute_taxes(
        self,
        current_ind_employee_income: np.ndarray,
        current_total_rent_paid: float,
        current_income_financial_assets: np.ndarray,
        current_ind_activity: np.ndarray,
        current_ind_realised_cons: np.ndarray,
        current_bank_profits: np.ndarray,
        current_firm_production: np.ndarray,
        current_firm_price: np.ndarray,
        current_firm_profits: np.ndarray,
        current_firm_industries: np.ndarray,
        current_household_new_real_wealth: np.ndarray,
        taxes_less_subsidies_rates: np.ndarray,
        current_total_exports: float,
    ) -> None:
        """Calculate all tax revenues for the current period.

        Computes revenues from multiple tax sources:
        - Production and VAT
        - Income and corporate taxes
        - Social insurance contributions
        - Capital formation and export taxes

        Args:
            current_ind_employee_income (np.ndarray): Employee incomes
            current_total_rent_paid (float): Total rent payments
            current_income_financial_assets (np.ndarray): Financial income
            current_ind_activity (np.ndarray): Individual activity status
            current_ind_realised_cons (np.ndarray): Consumption levels
            current_bank_profits (np.ndarray): Bank profits
            current_firm_production (np.ndarray): Firm production
            current_firm_price (np.ndarray): Product prices
            current_firm_profits (np.ndarray): Firm profits
            current_firm_industries (np.ndarray): Industry classifications
            current_household_new_real_wealth (np.ndarray): New wealth
            taxes_less_subsidies_rates (np.ndarray): Net tax rates
            current_total_exports (float): Total exports
        """
        # Taxes on production
        self.ts.taxes_production.append(
            [np.sum(taxes_less_subsidies_rates[current_firm_industries] * current_firm_production * current_firm_price)]
        )

        # Value-added taxes
        self.ts.taxes_vat.append([self.states["Value-added Tax"] * np.sum(current_ind_realised_cons)])

        # Taxes on capital formation
        self.ts.taxes_cf.append(
            [self.states["Capital Formation Tax"] * np.sum(np.maximum(0.0, current_household_new_real_wealth))]
        )

        # Corporate income taxes
        self.ts.taxes_corporate_income.append(
            [
                self.states["Profit Tax"]
                * (np.sum(np.maximum(current_firm_profits, 0)) + np.sum(np.maximum(current_bank_profits, 0)))
            ]
        )

        # Taxes on exports
        self.ts.taxes_exports.append([self.states["Export Tax"] * current_total_exports])

        # Total wages of employed individuals
        tot_wages_employed_ind = np.sum([current_ind_employee_income[current_ind_activity == ActivityStatus.EMPLOYED]])

        # Taxes on income
        self.ts.taxes_income.append(
            [
                self.states["Income Tax"] * (1 - self.states["Employee Social Insurance Tax"]) * tot_wages_employed_ind
                + self.states["Income Tax"] * current_total_rent_paid
                + self.states["Income Tax"] * current_income_financial_assets.sum(),
            ]
        )
        self.ts.taxes_rental_income.append([self.states["Income Tax"] * current_total_rent_paid])

        # Taxes on employer social insurance
        self.ts.taxes_employer_si.append([self.states["Employer Social Insurance Tax"] * tot_wages_employed_ind])

        # Taxes on employee social insurance
        self.ts.taxes_employee_si.append([self.states["Employee Social Insurance Tax"] * tot_wages_employed_ind])

    def compute_taxes_on_products(self) -> float:
        """Calculate total taxes on products and production.

        Aggregates various product-related taxes:
        - Production taxes
        - Value-added tax (VAT)
        - Capital formation tax
        - Export taxes

        Returns:
            float: Total tax revenue from products and production
        """
        return (
            self.ts.current("taxes_production")[0]
            + self.ts.current("taxes_vat")[0]
            + self.ts.current("taxes_cf")[0]
            + self.ts.current("taxes_exports")[0]
        )

    def compute_revenue(
        self,
        household_rent_paid_to_government: float,
    ) -> float:
        """Calculate total government revenue.

        Aggregates all revenue sources:
        - All tax revenues
        - Social insurance contributions
        - Rental income from public housing

        Args:
            household_rent_paid_to_government (float): Rent from public housing

        Returns:
            float: Total government revenue
        """
        self.ts.total_rent_received.append([household_rent_paid_to_government])
        return (
            self.ts.current("taxes_production")[0]
            + self.ts.current("taxes_vat")[0]
            + self.ts.current("taxes_cf")[0]
            + self.ts.current("taxes_corporate_income")[0]
            + self.ts.current("taxes_exports")[0]
            + self.ts.current("taxes_income")[0]
            + self.ts.current("taxes_employee_si")[0]
            + self.ts.current("taxes_employer_si")[0]
            + household_rent_paid_to_government
        )

    def compute_deficit(
        self,
        current_ind_activity: np.ndarray,
        current_household_social_transfers: np.ndarray,
        current_government_nominal_amount_spent: np.ndarray,
        government_interest_rates: float,
    ) -> np.ndarray:
        """Calculate the government deficit.

        Computes deficit as the difference between:
        Expenditures:
        - Unemployment benefits
        - Social transfers
        - Government spending
        - Interest payments
        And:
        - Total revenue

        Args:
            current_ind_activity (np.ndarray): Individual activity status
            current_household_social_transfers (np.ndarray): Social transfers
            current_government_nominal_amount_spent (np.ndarray): Spending
            government_interest_rates (float): Interest rate on debt

        Returns:
            np.ndarray: Government deficit (positive = deficit)
        """
        total_unemployment_benefits = (
            np.sum(current_ind_activity == ActivityStatus.UNEMPLOYED)
            * self.ts.current("unemployment_benefits_by_individual")[0]
        )
        total_household_social_transfers = np.sum(current_household_social_transfers)
        all_benefits = total_unemployment_benefits + total_household_social_transfers
        interest_payments = government_interest_rates * self.ts.current("debt")[0]
        return np.array(
            [
                all_benefits
                + np.sum(current_government_nominal_amount_spent)
                + interest_payments
                - self.ts.current("revenue")[0]
            ]
        )

    def compute_debt(self) -> np.ndarray:
        """Update government debt level.

        Calculates new debt level by adding current deficit
        to existing debt stock.

        Returns:
            np.ndarray: Updated government debt level
        """
        return np.array([self.ts.current("debt")[0] + self.ts.current("deficit")[0]])

    def save_to_h5(self, group: h5py.Group):
        """Save government data to HDF5 format.

        Stores all time series data in the specified HDF5 group.

        Args:
            group (h5py.Group): HDF5 group to save data in
        """
        self.ts.write_to_h5("central_government", group)

    def total_taxes(self):
        """Calculate total tax revenue on products.

        Returns:
            float: Aggregate tax revenue from all product-related taxes
        """
        return self.ts.get_aggregate("taxes_on_products")
