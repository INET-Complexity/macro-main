"""Debug utilities for investigating simulation behavior.

This module provides optional logging and diagnostic tools for investigating
specific hypotheses about model behavior. All debug features are opt-in and
should not affect normal simulation operation.
"""

from macromodel.debug.tfp_labor_logger import (
    TFPLaborLog,
    TFPLaborSnapshot,
    capture_tfp_labor_snapshot,
)

__all__ = [
    "TFPLaborLog",
    "TFPLaborSnapshot",
    "capture_tfp_labor_snapshot",
]
