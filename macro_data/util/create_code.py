"""
This module provides utilities for generating unique identifier codes.

The module implements a simple but effective method for generating numeric codes
that can be used as unique identifiers in economic data structures. The codes
are purely numeric to ensure compatibility with various data storage and
processing systems.

Key features:
- Fixed-length numeric codes
- Random generation for uniqueness
- Simple and efficient implementation
- Suitable for economic entity identification

Example:
    ```python
    from macro_data.util.create_code import create_code

    # Generate unique identifiers for firms
    firm_ids = [create_code() for _ in range(3)]
    # Example output: ['123456789012', '234567890123', '345678901234']
    ```
"""

import random
import string


def create_code() -> str:
    """
    Generate a 12-digit random numeric code.

    This function creates a random numeric string suitable for use as an
    identifier in economic data structures. The code is:
    - Always 12 digits long
    - Composed only of numeric characters (0-9)
    - Generated randomly for high probability of uniqueness

    The function is particularly useful for:
    - Creating firm identification numbers
    - Generating transaction reference codes
    - Creating unique account numbers
    - Identifying economic entities

    Returns:
        str: A 12-character string containing only digits.

    Notes:
        - The function uses the system's random number generator
        - While collisions are possible, they are unlikely given the
          12-digit length (10^12 possible values)
        - The codes are not cryptographically secure and should not
          be used for security purposes

    Example:
        ```python
        # Generate a single code
        id_code = create_code()  # e.g., '123456789012'

        # Generate multiple unique codes
        entity_codes = [create_code() for _ in range(5)]
        ```
    """
    return "".join(random.choice(string.digits) for _ in range(12))
