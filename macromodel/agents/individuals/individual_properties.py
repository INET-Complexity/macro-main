"""Individual demographic and socioeconomic characteristics.

This module defines the fundamental properties and classifications of individuals
in the macroeconomic model through enums that capture:
- Economic activity status and employment
- Demographic characteristics
- Educational attainment levels

These properties are used to:
- Track individual status changes
- Group individuals into households
- Calculate economic outcomes
- Analyze population dynamics
"""

from enum import Enum


class ActivityStatus(Enum):
    """Individual's economic activity and employment status.

    This enum classifies individuals based on their participation in
    the labor market and investment activities:
    - Employment status (employed/unemployed)
    - Economic activity (active/inactive)
    - Investment roles (firm/bank investor)

    These statuses affect:
    - Income generation
    - Labor market participation
    - Investment returns
    - Economic contributions
    """

    EMPLOYED = 1  # Currently employed in labor market
    UNEMPLOYED = 2  # Actively seeking employment
    NOT_ECONOMICALLY_ACTIVE = 3  # Not participating in labor market
    FIRM_INVESTOR = 4  # Holds investments in firms
    BANK_INVESTOR = 5  # Holds investments in banks


class Gender(Enum):
    """Individual's gender classification.

    This enum captures binary gender categories used for:
    - Demographic analysis
    - Population statistics
    - Labor market segmentation
    - Policy impact assessment
    """

    MALE = 1
    FEMALE = 2


class Education(Enum):
    """Individual's highest educational attainment level.

    This enum represents the educational hierarchy from no formal
    education through doctoral studies. Used for:
    - Human capital assessment
    - Wage determination
    - Labor market matching
    - Skill level classification
    - Productivity analysis
    """

    NONE = 0  # No formal education
    PRIMARY = 1  # Primary education
    LOWER_SECONDARY = 2  # Lower secondary education
    UPPER_SECONDARY = 3  # Upper secondary education
    POST_SECONDARY = 4  # Post-secondary non-tertiary
    SHORT_TERTIARY = 5  # Short-cycle tertiary
    BACHELOR = 6  # Bachelor's or equivalent
    MASTER = 7  # Master's or equivalent
    DOCTORAL = 8  # Doctoral or equivalent
