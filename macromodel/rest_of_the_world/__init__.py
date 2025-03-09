"""Rest of the World (ROW) package.

This package implements the external sector component of the macroeconomic model,
representing all economies outside the explicitly modeled countries. It provides:

1. Core Components:
   - RestOfTheWorld class for external sector representation
   - Time series management for ROW variables
   - International trade flow handling

2. Economic Functions:
   - Import demand determination
   - Export supply decisions
   - Price setting mechanisms
   - Inflation dynamics
   - Excess demand management

3. Market Interactions:
   - Trade flow adjustments
   - Price level convergence
   - Market clearing processes
   - Exchange rate effects

The package serves as the closure for international trade in the model,
ensuring consistent global accounting and economic relationships.
"""

from .rest_of_the_world import RestOfTheWorld

__all__ = ["RestOfTheWorld"]
