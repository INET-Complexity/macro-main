"""Loan type definitions for the credit market.

This module defines the different types of loans available in the credit market system.
Each loan type has specific characteristics and is handled differently during:
- Market clearing
- Risk assessment
- Interest rate determination
- Payment processing
- Default handling

The loan types are divided into two main categories:

1. Firm Loans:
   - Short-term loans: Used for working capital and operational expenses
     * Shorter duration (typically < 1 year)
     * Higher interest rates
     * More frequent payments
   - Long-term loans: Used for capital investment and expansion
     * Longer duration (typically > 1 year)
     * Lower interest rates
     * Collateral often required

2. Household Loans:
   - Consumption loans: Personal loans for goods and services
     * Medium duration
     * Higher interest rates
     * Unsecured lending
   - Mortgages: Home purchase and refinancing
     * Longest duration
     * Lower interest rates
     * Secured by property
"""

from enum import Enum


class LoanTypes(Enum):
    """Enumeration of available loan types in the credit market.

    Each loan type represents a distinct lending product with its own:
    - Target borrower (firms vs households)
    - Purpose (working capital, investment, consumption, housing)
    - Risk profile
    - Typical terms and conditions

    Attributes:
        FIRM_SHORT_TERM_LOAN (int): Working capital and operational loans to firms
        FIRM_LONG_TERM_LOAN (int): Capital investment loans to firms
        HOUSEHOLD_CONSUMPTION_LOAN (int): Personal loans to households
        MORTGAGE (int): Home purchase/refinance loans to households

    Example:
        >>> loan_type = LoanTypes.MORTGAGE
        >>> if loan_type == LoanTypes.MORTGAGE:
        ...     interest_rate = base_rate + mortgage_spread
    """

    FIRM_SHORT_TERM_LOAN: int = 1
    FIRM_LONG_TERM_LOAN: int = 2
    HOUSEHOLD_CONSUMPTION_LOAN: int = 3
    MORTGAGE: int = 4
