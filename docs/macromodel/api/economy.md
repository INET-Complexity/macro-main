# Economy API Reference

This section documents the Economy entity in the macromodel package, which manages and tracks aggregate economic metrics, market-level interactions, and macroeconomic indicators for each country.

::: macromodel.economy.economy.Economy
    options:
        members:
            - Economy
            - from_agents
            - reset
            - set_estimates
            - compute_inflation
            - record_global_trade
            - compute_rental_market_aggregates
            - compute_gdp
            - total_exports
            - total_cpi_inflation
            - total_ppi_inflation
            - total_cfpi_inflation
            - unemployment_rate
            - gdp_expenditure
            - gdp_output
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

### Sentiment

::: macromodel.economy.func.sentiment.SentimentSetter
    options:
        members:
            - SentimentSetter
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

### Inflation Forecasting

::: macromodel.economy.func.inflation.InflationForecasting
    options:
        members:
            - InflationForecasting
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

### Growth Forecasting

::: macromodel.economy.func.growth.GrowthForecasting
    options:
        members:
            - GrowthForecasting
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
