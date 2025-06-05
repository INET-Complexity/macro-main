# Banks API Reference

This section documents the Banks agent in the macromodel package, which implements the banking sector and its financial intermediation functions.

::: macromodel.agents.banks.banks.Banks
    options:
        members:
            - Banks
            - from_pickled_agent
            - reset
            - compute_estimated_profits
            - set_interest_rates
            - compute_interest_received_on_deposits
            - compute_profits
            - update_deposits
            - update_loans
            - compute_market_share
            - compute_equity
            - compute_liability
            - compute_deposits
            - handle_insolvency
            - compute_insolvency_rate
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

## Functions

### Bank Demography

::: macromodel.agents.banks.func.demography.BankDemography
    options:
        members:
            - BankDemography
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

### Interest Rate Setting

::: macromodel.agents.banks.func.interest_rates.InterestRatesSetter
    options:
        members:
            - InterestRatesSetter
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

### Profit Estimation

::: macromodel.agents.banks.func.profit_estimator.BankProfitsSetter
    options:
        members:
            - BankProfitsSetter
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
