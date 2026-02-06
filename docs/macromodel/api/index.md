# Macromodel API Reference

This section provides an overview and links to the main API documentation for the macromodel package.

## Core Components

- [Country](country.md): Country-level aggregation and parameter management
- [Rest of World](rest_of_world.md): External sector and international linkages
- [Economy](economy.md): Economy-wide coordination and simulation
- [Agents](agents/index.md): Economic agents (households, firms, banks, government, etc.)

## Markets

- [Goods Market](markets/goods_market.md): Goods market mechanisms and clearing
- [Credit Market](markets/credit_market.md): Credit and lending mechanisms
- [Housing Market](markets/housing_market.md): Housing and property market
- [Labour Market](markets/labour_market.md): Labor market and employment

## Main API Sections

- [Agents](agents/index.md): All economic agent classes and their behaviors
- [Country](country.md): Country-level aggregation and parameters
- [Economy](economy.md): System-wide coordination and macro indicators
- [Simulation](simulation.md): Simulation engine and time-stepping

Each section provides detailed documentation for the relevant classes, methods, and functions.

## Simulation

::: macromodel.simulation.Simulation
    options:
        members:
            - Simulation
            - from_datawrapper
            - reset
            - iterate
            - run
            - save
            - shallow_df_dict
            - shallow_hdf_save
            - get_country_shallow_output
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
