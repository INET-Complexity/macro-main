"""Time series management for Central Government agent.

This module handles the creation and management of time series data
for the central government agent, including:
- Fiscal variables (revenue, deficit, debt)
- Tax collections by type
- Social benefits and transfers
- Public housing income

The time series provide historical tracking of:
- Government financial position
- Tax revenue streams
- Social benefit payments
- Public sector operations
"""

import numpy as np
import pandas as pd

from macromodel.timeseries import TimeSeries


def create_central_government_timeseries(
    data: pd.DataFrame,
    number_of_unemployed_individuals: int,
) -> TimeSeries:
    """Create time series objects for central government variables.

    Initializes time series tracking for:
    - Fiscal position (debt, deficit, revenue)
    - Tax collections by type
    - Social benefits and transfers
    - Public housing income

    Args:
        data (pd.DataFrame): Initial government data including historical
            values for all tracked variables
        number_of_unemployed_individuals (int): Count of unemployed people
            for per-person benefit calculation

    Returns:
        TimeSeries: Initialized time series containing all government
            variables with their initial values
    """
    return TimeSeries(
        debt=np.array([float(data["Debt"].iloc[0])]),
        unemployment_benefits_by_individual=[
            data["Total Unemployment Benefits"].values[0] / number_of_unemployed_individuals
        ],
        total_other_benefits=[data["Other Social Benefits"].values[0]],
        #
        taxes_production=[data["Taxes on Production"].values[0]],
        taxes_vat=[data["VAT"].values[0]],
        taxes_cf=[data["Capital Formation Taxes"].values[0]],
        taxes_corporate_income=[data["Corporate Taxes"].values[0]],
        taxes_exports=[data["Export Taxes"].values[0]],
        taxes_income=[data["Income Taxes"].values[0]],
        taxes_rental_income=[data["Rental Income Taxes"].values[0]],
        taxes_employee_si=[data["Employee SI Tax"].values[0]],
        taxes_employer_si=[data["Employer SI Tax"].values[0]],
        taxes_on_products=[data["Taxes on Products"].values[0]],
        total_rent_received=[data["Total Social Housing Rent"].values[0]],
        #
        revenue=[data["Revenue"].values[0]],
        deficit=np.array([np.nan]),
        #
        bank_equity_injection=[data["Bank Equity Injection"].values[0]],
    )
