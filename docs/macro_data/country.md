# Country Object

The `Country` object is an enumeration (enum) of countries using ISO 3166-1 alpha-3 codes. It provides a type-safe way to work with country codes throughout the macro_data package and the broader model.

## Features

- Enum values are standard 3-letter ISO country codes (e.g., 'FRA', 'DEU').
- Includes all recognized countries and a special `ROW` (Rest of World) designation.
- Provides EU membership status via the `is_eu_country` property.
- Utilities for converting between 2-letter and 3-letter codes.
- String representation matches the country code or name.

## Example Usage

```python
from macro_data.configuration.countries import Country

# Create a country instance
france = Country.FRA

# Check if it's an EU country
is_eu = france.is_eu_country  # True

# Convert codes
two_letter = france.to_two_letter_code()  # 'FR'
three_letter = Country.convert_two_letter_to_three('FR')  # 'FRA'
```

## API Reference

::: macro_data.configuration.countries.Country
    options:
        members:
            - is_eu_country
            - to_two_letter_code
            - convert_two_letter_to_three
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
