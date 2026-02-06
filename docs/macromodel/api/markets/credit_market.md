# Credit Market API Reference

This section documents the Credit Market module, which manages lending, loan servicing, and credit allocation between banks, firms, and households.

::: macromodel.markets.credit_market.credit_market.CreditMarket
    options:
        members:
            - CreditMarket
            - from_pickled_market
            - from_data
            - reset
            - clear
            - pay_firm_installments
            - pay_household_installments
            - remove_repaid_loans
            - compute_aggregates
            - compute_outstanding_short_term_loans_by_firm
            - compute_outstanding_long_term_loans_by_firm
            - compute_outstanding_consumption_loans_by_household
            - compute_outstanding_mortgages_by_household
            - compute_outstanding_loans_by_bank
            - compute_interest_paid_by_firm
            - compute_interest_paid_by_household
            - compute_interest_received_by_bank
            - remove_loans_to_firm
            - remove_loans_to_households
            - remove_loans_by_bank
            - save_to_h5
        show_root_heading: true
        show_signature_annotations: true
        show_docstring: true
        show_source: false
        show_bases: false
        show_inheritance_diagram: false
        show_if_no_docstring: true
        heading_level: 4
        show_module_name: false
        hide_name: false

## Loan Types

::: macromodel.markets.credit_market.types_of_loans.LoanTypes
    options:
        members:
            - LoanTypes
        show_root_heading: true
        show_signature_annotations: true
        show_docstring: true
        show_source: false
        show_bases: false
        show_inheritance_diagram: false
        show_if_no_docstring: true
        heading_level: 4
        show_module_name: false
        hide_name: false

## Clearing Mechanisms (Abstract)

::: macromodel.markets.credit_market.func.clearing.CreditMarketClearer
    options:
        members:
            - CreditMarketClearer
        show_root_heading: true
        show_signature_annotations: true
        show_docstring: true
        show_source: false
        show_bases: false
        show_inheritance_diagram: false
        show_if_no_docstring: true
        heading_level: 4
        show_module_name: false
        hide_name: false
