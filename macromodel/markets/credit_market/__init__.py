"""Credit market package for macroeconomic agent-based model.

This package implements a sophisticated credit market system that manages lending
relationships between financial institutions and borrowers. It includes:

Core Components:
1. CreditMarket: Main market implementation
   - Market clearing mechanism
   - Loan lifecycle management
   - Default handling
   - State tracking

2. Loan Types:
   - Short-term firm loans
   - Long-term firm loans
   - Consumer loans
   - Mortgages

3. Market Functions:
   - Clearing algorithms
   - Interest rate determination
   - Risk assessment
   - Credit allocation

4. Time Series:
   - Loan volume tracking
   - Performance metrics
   - Market statistics

The package provides a flexible framework for modeling credit markets with:
- Multiple types of financial institutions
- Different borrower classes (firms, households)
- Various loan products
- Risk-based pricing
- Regulatory constraints
"""

from .credit_market import CreditMarket
from .types_of_loans import LoanTypes

__all__ = ["CreditMarket", "LoanTypes"]
