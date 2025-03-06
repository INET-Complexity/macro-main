"""Household demographic properties and classifications.

This module defines household types and demographic characteristics
through:
- Household composition categories
- Adult count tracking
- Child presence indicators
- Age group classifications

The implementation handles:
- Multi-adult households
- Single-parent families
- Child-present households
- Age-based categorization
"""

from enum import Enum


class HouseholdType(Enum):
    """Classification of households by composition and demographics.

    This enum defines household categories based on:
    - Number of adults
    - Presence of children
    - Age of adults
    - Family structure

    The categories consider:
    - Single vs multi-adult households
    - Age thresholds (65 years)
    - Number of children
    - Parent status
    """

    TWO_ADULTS_YOUNGER_THAN_65 = 6
    TWO_ADULTS_ONE_AT_LEAST_65 = 7
    THREE_OR_MORE_ADULTS = 8
    SINGLE_PARENT_WITH_CHILDREN = 9
    TWO_ADULTS_WITH_ONE_CHILD = 10
    TWO_ADULTS_WITH_TWO_CHILDREN = 11
    TWO_ADULTS_WITH_AT_LEAST_THREE_CHILDREN = 12
    THREE_OR_MORE_ADULTS_WITH_CHILDREN = 13
    ONE_ADULT_YOUNGER_THAN_64 = 51
    ONE_ADULT_OLDER_THAN_65 = 52
