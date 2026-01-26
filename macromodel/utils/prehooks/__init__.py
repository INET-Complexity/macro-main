"""Pre-hooks for simulation interventions.

This module provides factory functions to create pre-hooks that can be registered
with a Simulation object to perform interventions at specific timesteps.
"""

from macromodel.utils.prehooks.productivity_subsidy import (
    create_productivity_subsidy_hook,
)

__all__ = ["create_productivity_subsidy_hook"]
