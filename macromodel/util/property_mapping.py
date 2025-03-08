"""Enumeration mapping utilities for property types.

This module provides utilities for mapping between numerical values and
enumeration types, particularly useful for converting between stored
property values and their semantic representations in the model.

The module supports:
- Enum value mapping
- Array-based conversion
- Type-safe property handling
"""

from enum import Enum, EnumMeta

import numpy as np


def map_to_enum(values: np.ndarray, property_enum: EnumMeta) -> np.ndarray[Enum]:
    """Map numerical values to their corresponding enum members.

    This function converts an array of numerical values to an array
    of enumeration members, useful for converting stored property
    values to their semantic representations.

    Args:
        values: Array of numerical values to convert
        property_enum: Enumeration class to map values to

    Returns:
        np.ndarray[Enum]: Array of enum members corresponding to input values

    Example:
        status_enum = map_to_enum(
            status_values,
            ActivityStatus
        )
    """
    enum_map = {p.value: p for p in property_enum}
    return list(map(enum_map.get, values))
