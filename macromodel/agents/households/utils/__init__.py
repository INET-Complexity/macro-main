"""Household utilities module.

This module provides utility functions for household agent operations including:
- Bundle matrix creation for substitution patterns
- Consumption weight calculations
- Configuration helpers
"""

from .create_bundle_matrix import create_bundle_matrix

__all__ = ["create_bundle_matrix"]