# Region Object

The `Region` object is a string subclass that represents a region code with additional metadata. It is used to represent subnational regions within a country, supporting hierarchical modeling and aggregation.

## Features

- Stores a region code, parent country, region name, and value-added ratio (`va_ratio`).
- Provides properties for accessing the parent country, region name, EU membership, and value-added ratio.
- Supports creation from a code and name, and is compatible with Pydantic for configuration validation.

## Example Usage

```python
from macro_data.configuration.region import Region
from macro_data.configuration.countries import Country

# Create a region instance
region = Region("FRA_IDF", Country.FRA, "Île-de-France", va_ratio=0.25)

# Access metadata
parent = region.parent_country  # Country.FRA
name = region.name  # 'Île-de-France'
is_eu = region.is_eu_country  # True
va = region.va_ratio  # 0.25
```

## API Reference

::: macro_data.configuration.region.Region
    options:
        members:
            - parent_country
            - name
            - is_eu_country
            - va_ratio
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
