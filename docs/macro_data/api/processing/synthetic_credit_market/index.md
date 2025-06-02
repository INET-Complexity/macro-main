# SyntheticCreditMarket

The `SyntheticCreditMarket` class is a container for preprocessed credit market relationship data between financial institutions and borrowers. It organizes initial loan states and parameters for model initialization.

## Core Functionality

The class handles:
1. Credit Relationship Data:
   - Bank-firm loan relationships
   - Bank-household loan relationships
   - Initial loan parameters

2. Loan Type Organization:
   - Long-term firm loans
   - Short-term firm loans
   - Consumer loans
   - Payday loans
   - Mortgage loans

3. Loan Parameter Processing:
   - Principal amounts
   - Interest rates
   - Installment calculations
   - Maturity periods

## Key Attributes

- `country_name`: Country identifier for data collection
- `year`: Reference year for preprocessing
- `longterm_loans`: Preprocessed firm long-term loan data
- `shortterm_loans`: Preprocessed firm short-term loan data
- `consumption_expansion_loans`: Preprocessed consumer loan data
- `payday_loans`: Preprocessed payday loan data
- `mortgage_loans`: Preprocessed mortgage loan data

## Factory Methods

The class provides a factory method `create_from_agents` that creates a `SyntheticCreditMarket` instance by:
1. Processing credit relationship data from various economic agents
2. Matching borrowers with corresponding banks
3. Calculating initial loan parameters
4. Organizing data into loan type categories

# Implementation

::: macro_data.processing.synthetic_credit_market.synthetic_credit_market
    options:
        members:
            - SyntheticCreditMarket
            - create_from_agents

# Loan Data Classes

The credit market module includes several specialized classes for different types of loans:

## LoanData

Base class for preprocessing loan-specific credit data. It provides:
- Principal amounts by bank-borrower pair
- Interest amounts by bank-borrower pair
- Installment amounts by bank-borrower pair

## LongtermLoans

Container for preprocessed long-term firm loan data:
- Principal amounts from firm debt data
- Interest amounts using bank long-term rates
- Installment amounts based on maturity

## ShorttermLoans

Container for preprocessed short-term firm loan data:
- Principal amounts from firm debt data
- Interest amounts using bank short-term rates
- Installment amounts based on maturity

## ConsumptionExpansionLoans

Container for preprocessed consumer loan data:
- Principal amounts from household debt data
- Interest amounts using bank consumer rates
- Installment amounts based on maturity

## PaydayLoans

Container for preprocessed payday loan data:
- Principal amounts from household data
- Interest amounts using bank payday rates
- Installment amounts based on maturity

## MortgageLoans

Container for preprocessed mortgage loan data:
- Principal amounts from household mortgage data
- Interest amounts using bank mortgage rates
- Installment amounts based on maturity

# Implementation

::: macro_data.processing.synthetic_credit_market.loan_data
    options:
        members:
            - LoanData
            - LongtermLoans
            - ShorttermLoans
            - ConsumptionExpansionLoans
            - PaydayLoans
            - MortgageLoans 