"""
Module containing mappings between OECD industry codes and internal sector indices.

This module provides a standardized mapping between OECD industry classification codes
and internal sector indices used in the macro model. The OECD codes are based on
ISIC Rev. 4 divisions, where ranges (e.g., '05_09') indicate inclusive spans of
division codes.

Industry Sectors:
    1: Mining and quarrying (05-09)
    2: Manufacturing (10-33)
    3: Electricity, gas, steam and air conditioning supply (35)
    4: Water supply, sewerage, waste management (36-39)
    5: Construction (41-43)
    6: Wholesale and retail trade (45-47)
    7: Transportation and storage (49-53)
    8: Accommodation and food service activities (55-56)
    9: Information and communication (58-63)
    10: Financial and insurance activities (64-66)
    11: Real estate activities (68)
    12: Professional, scientific and technical activities (69-75)
    13: Administrative and support service activities (77-82)

Example:
    ```python
    from macro_data.readers.economic_data.oecd_mappings import INDUSTRY_MAPPING

    # Get internal sector index for manufacturing
    manufacturing_sector = INDUSTRY_MAPPING['10_33']  # Returns 2
    ```
"""

INDUSTRY_MAPPING = {
    "05_09": 1,
    "10_33": 2,
    "35": 3,
    "36_39": 4,
    "41_43": 5,
    "45_47": 6,
    "49_53": 7,
    "55_56": 8,
    "58_63": 9,
    "64_66": 10,
    "68": 11,
    "69_75": 12,
    "77_82": 13,
}
