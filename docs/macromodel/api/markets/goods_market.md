# Goods Market API Reference

This section documents the Goods Market module, which manages transactions, market clearing, and supply chains for goods across industries and countries.

::: macromodel.markets.goods_market.goods_market.GoodsMarket
    options:
        members:
            - GoodsMarket
            - from_data
            - reset
            - prepare
            - clear
            - record
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

## Value Types

::: macromodel.markets.goods_market.value_type.ValueType
    options:
        members:
            - ValueType
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

::: macromodel.markets.goods_market.func.clearing.GoodsMarketClearer
    options:
        members:
            - GoodsMarketClearer
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
