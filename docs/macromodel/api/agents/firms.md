# Firms API Reference

This section documents the Firms agent in the macromodel package, which represents the productive sector of the economy and manages firm-level production, pricing, investment, and financial decisions.

The Firms agent includes comprehensive support for productivity dynamics through:
- **Productivity Investment Planning**: Firms can strategically invest in productivity improvements based on expected returns
- **Total Factor Productivity (TFP) Growth**: Multiple models for how productivity evolves over time, including deterministic, stochastic, and sector-specific growth patterns
- **Endogenous Growth**: Productivity improvements driven by firm-level investment decisions, creating a feedback loop between profitability and productivity

The Firms agent also supports flexible production structures through:
- **Substitution Bundles**: Groups of inputs that can be substituted for one another using CES (Constant Elasticity of Substitution) production functions
- **Bundled Leontief Production**: Combines fixed-proportion Leontief technology for most inputs with CES substitution within specified bundles (e.g., energy inputs)
- **Dynamic Input Mix**: Firms adjust their input combinations based on relative prices, allowing for realistic responses to price shocks and policy changes
- **Sector-Specific Configurations**: Different industries can have different substitution possibilities, reflecting technological constraints

::: macromodel.agents.firms.firms.Firms
    options:
        members:
            - Firms
            - from_pickled_agent
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

### Demography

::: macromodel.agents.firms.func.demography.FirmDemography
    options:
        members:
            - FirmDemography
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

### Target Production

::: macromodel.agents.firms.func.target_production.TargetProductionSetter
    options:
        members:
            - TargetProductionSetter
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

### Labour Productivity

::: macromodel.agents.firms.func.labour_productivity.LabourProductivitySetter
    options:
        members:
            - LabourProductivitySetter
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

### Wage Setting

::: macromodel.agents.firms.func.wage_setter.FirmWageSetter
    options:
        members:
            - FirmWageSetter
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

### Price Setting

::: macromodel.agents.firms.func.prices.PriceSetter
    options:
        members:
            - PriceSetter
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

### Production

::: macromodel.agents.firms.func.production.ProductionSetter
    options:
        members:
            - ProductionSetter
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

### Excess Demand

::: macromodel.agents.firms.func.excess_demand.ExcessDemandSetter
    options:
        members:
            - ExcessDemandSetter
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

### Desired Labour

::: macromodel.agents.firms.func.desired_labour.DesiredLabourSetter
    options:
        members:
            - DesiredLabourSetter
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

### Productivity Investment Planning

::: macromodel.agents.firms.func.productivity_investment_planner.ProductivityInvestmentPlanner
    options:
        members:
            - ProductivityInvestmentPlanner
            - NoOpProductivityInvestmentPlanner
            - SimpleProductivityInvestmentPlanner
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

### Productivity Growth (TFP)

::: macromodel.agents.firms.func.productivity_growth.ProductivityGrowth
    options:
        members:
            - ProductivityGrowth
            - NoOpTFPGrowth
            - SimpleTFPGrowth
            - StochasticTFPGrowth
            - SectoralTFPGrowth
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
