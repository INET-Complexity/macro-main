"""
This module provides industry mapping dictionaries for both WIOD (World Input-Output Database)
and ICIO (Inter-Country Input Output) tables. It defines the relationships between detailed
industry codes and their aggregated sectors, enabling flexible analysis at different levels
of granularity.

The module supports two main data sources:
1. WIOD (World Input-Output Database)
2. ICIO (OECD Inter-Country Input Output Tables)

For each data source, two types of mappings are provided:
1. AGGREGATE: Maps detailed industry codes to their high-level sectors
2. ALL: Maintains detailed industry codes while providing a consistent structure

Each mapping has an inverse version (e.g., WIOD_AGGREGATE_INV) that maps from
detailed codes to their parent sectors, useful for reverse lookups.

Example:
    ```python
    from macro_data.readers.io_tables.mappings import ICIO_AGGREGATE, ICIO_ALL

    # Using aggregate mapping
    manufacturing_subsectors = ICIO_AGGREGATE['C']  # Get all manufacturing industries

    # Using detailed mapping
    chemical_sector = ICIO_ALL['C20']  # Get chemical industry details
    ```
"""


def sub_to_aggregate(dictio: dict[str, list[str]]) -> dict[str, str]:
    """
    Create an inverse mapping from sub-sectors to their aggregate sectors.

    This utility function inverts a mapping dictionary, creating a new dictionary
    where each sub-sector maps to its parent aggregate sector. This is useful for
    quickly determining which aggregate sector a detailed industry code belongs to.

    Args:
        dictio (dict[str, list[str]]): Original mapping where keys are aggregate
            sectors and values are lists of sub-sectors

    Returns:
        dict[str, str]: Inverted mapping where keys are sub-sectors and values
            are their corresponding aggregate sectors

    Example:
        ```python
        original = {'C': ['C10', 'C11', 'C12']}
        inverse = sub_to_aggregate(original)
        # Result: {'C10': 'C', 'C11': 'C', 'C12': 'C'}
        ```
    """
    new_dict = {}
    for aggregate_sector, sub_sectors in dictio.items():
        for sub_sector in sub_sectors:
            new_dict[sub_sector] = aggregate_sector
    return new_dict


# WIOD (World Input-Output Database) industry mappings

# Maps detailed WIOD industry codes to their aggregate sectors
WIOD_AGGREGATE = {
    "A": ["A01", "A02", "A03"],
    "B": ["B"],
    "C": [
        "C10-C12",
        "C13-C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "C21",
        "C22",
        "C23",
        "C24",
        "C25",
        "C26",
        "C27",
        "C28",
        "C29",
        "C30",
        "C31_C32",
        "C33",
    ],
    "D": ["D35"],
    "E": ["E36", "E37-E39"],
    "F": ["F"],
    "G": ["G45", "G46", "G47"],
    "H": ["H49", "H50", "H51", "H52", "H53"],
    "I": ["I"],
    "J": ["J58", "J59_J60", "J61", "J62_J63"],
    "K": ["K64", "K65", "K66"],
    "L": ["L68"],
    "M": ["M69_M70", "M71", "M72", "M73", "M74_M75"],
    "N": ["N"],
    "O": ["O84"],
    "P": ["P85"],
    "Q": ["Q"],
    "R_S": ["R_S"],
    "T": ["T"],
    "U": ["U"],
    "CONS_h": ["CONS_h"],
    "CONS_np": ["CONS_np"],
    "CONS_g": ["CONS_g"],
    "GFCF": ["GFCF"],
    "INVEN": ["INVEN"],
    "GO": ["GO"],
    "II_fob": ["II_fob"],
    "TXSP": ["TXSP"],
    "EXP_adj": ["EXP_adj"],
    "VA": ["VA"],
    "IntTTM": ["IntTTM"],
}

# Inverse mapping from detailed to aggregate WIOD codes
WIOD_AGGREGATE_INV = sub_to_aggregate(WIOD_AGGREGATE)

# Detailed WIOD industry mapping maintaining full granularity
WIOD_ALL = {
    "A01": ["A01"],
    "A03": ["A03"],
    "B": ["B"],
    "B05": ["B05"],
    "B07": ["B07"],
    "B09": ["B09"],
    "C10T12": ["C10-C12"],
    "C13T15": ["C13-C15"],
    "C16": ["C16"],
    "C17": ["C17"],
    "C19": ["C19"],
    "C20": ["C20"],
    "C21": ["C21"],
    "C22": ["C22"],
    "C23": ["C23"],
    "C24": ["C24"],
    "C25": ["C25"],
    "C26": ["C26"],
    "C27": ["C27"],
    "C28": ["C28"],
    "C29": ["C29"],
    "C30": ["C30"],
    "C31T33": ["C31_C32", "C33"],
    "D": ["D35"],
    "E": ["E36", "E37-E39"],
    "F": ["F"],
    "G": ["G45", "G46", "G47"],
    "H49": ["H49"],
    "H50": ["H50"],
    "H51": ["H51"],
    "H52": ["H52"],
    "H53": ["H53"],
    "I": ["I"],
    "J58T60": ["J58", "J59_J60"],
    "J61": ["J61"],
    "J62": ["J62_J63"],
    "K": ["K64", "K65", "K66"],
    "L": ["L68"],
    "M": ["M69_M70", "M71", "M72", "M73", "M74_M75"],
    "N": ["N"],
    "O": ["O84"],
    "P": ["P85"],
    "Q": ["Q"],
    "R_S": ["R_S"],
    "T": ["T"],
    "U": ["U"],
}

# Inverse mapping from detailed to aggregate WIOD codes (full granularity)
WIOD_ALL_INV = sub_to_aggregate(WIOD_ALL)

# ICIO (OECD Inter-Country Input Output) industry mappings

# Maps detailed ICIO industry codes to their aggregate sectors
ICIO_AGGREGATE = {
    "A": ["A01", "A03"],
    "B": ["B05", "B07", "B09"],
    "C": [
        "C10T12",
        "C13T15",
        "C16",
        "C17",
        "C19",
        "C20",
        "C21",
        "C22",
        "C23",
        "C24",
        "C25",
        "C26",
        "C27",
        "C28",
        "C29",
        "C30",
        "C31T33",
    ],
    "D": ["D"],
    "E": ["E"],
    "F": ["F"],
    "G": ["G"],
    "H": ["H49", "H50", "H51", "H52", "H53"],
    "I": ["I"],
    "J": ["J58T60", "J61", "J62"],
    "K": ["K"],
    "L": ["L"],
    "M": ["M"],
    "N": ["N"],
    "O": ["O"],
    "P": ["P"],
    "Q": ["Q"],
    "R_S": ["R", "S", "T", "R_S"],
    "Fixed Capital Formation": ["GFCF", "INVNT"],
    "Government Consumption": ["GGFC"],
    "Household Consumption": ["HFCE", "NPISH", "DPABR"],
    "Taxes Less Subsidies": ["TLS"],
    "Value Added": ["VA"],
    "Gross Output": ["OUT"],
}

# Inverse mapping from detailed to aggregate ICIO codes
ICIO_AGGREGATE_INV = sub_to_aggregate(ICIO_AGGREGATE)

# Detailed ICIO industry mapping maintaining full granularity
ICIO_ALL = {
    "A01": ["A01"],
    "A03": ["A03"],
    "B05": ["B05"],
    "B07": ["B07"],
    "B09": ["B09"],
    "C10T12": ["C10T12"],
    "C13T15": ["C13T15"],
    "C16": ["C16"],
    "C17": ["C17"],
    "C19": ["C19"],
    "C20": ["C20"],
    "C21": ["C21"],
    "C22": ["C22"],
    "C23": ["C23"],
    "C24": ["C24"],
    "C25": ["C25"],
    "C26": ["C26"],
    "C27": ["C27"],
    "C28": ["C28"],
    "C29": ["C29"],
    "C30": ["C30"],
    "C31T33": ["C31T33"],
    "D": ["D"],
    "E": ["E"],
    "F": ["F"],
    "G": ["G"],
    "H49": ["H49"],
    "H50": ["H50"],
    "H51": ["H51"],
    "H52": ["H52"],
    "H53": ["H53"],
    "J58T60": ["J58T60"],
    "J61": ["J61"],
    "J62": ["J62"],
    "I": ["I"],
    "K": ["K"],
    "L": ["L"],
    "M": ["M"],
    "N": ["N"],
    "O": ["O"],
    "P": ["P"],
    "Q": ["Q"],
    "R_S": ["R", "S", "T"],
    "Fixed Capital Formation": ["GFCF", "INVNT"],
    "Government Consumption": ["GGFC"],
    "Household Consumption": ["HFCE", "NPISH", "DPABR"],
    "Taxes Less Subsidies": ["TLS"],
    "Value Added": ["VA"],
    "Gross Output": ["OUT"],
}

# Inverse mapping from detailed to aggregate ICIO codes (full granularity)
ICIO_ALL_INV = sub_to_aggregate(ICIO_ALL)
