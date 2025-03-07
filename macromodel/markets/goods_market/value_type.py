"""Value type definitions for goods market transactions.

This module defines the possible value types for goods market transactions,
distinguishing between real quantities, nominal values, and undefined states.
These types are used throughout the goods market clearing process to ensure
proper handling of different transaction value representations.
"""

from enum import Enum


class ValueType(Enum):
    """Value type enumeration for goods market transactions.

    Defines the possible value types for market transactions:
    - REAL: Physical quantities or real values
    - NOMINAL: Monetary values in current prices
    - NONE: Undefined or not applicable state
    """

    REAL = 0
    NOMINAL = 1
    NONE = 2
