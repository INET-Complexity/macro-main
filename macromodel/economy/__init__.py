"""Economy module.

This module implements tracking and computation of macroeconomic indicators
and aggregates. Key features include:

1. Economic Measurement:
   - Price indices and inflation
   - GDP components and growth
   - Labor market metrics
   - Financial conditions

2. Market Integration:
   - Cross-market consistency
   - Flow of funds tracking
   - Stock-flow relationships
   - Sectoral balances

3. Economic Analysis:
   - Growth decomposition
   - Price dynamics
   - Market conditions
   - International flows

The economy module serves as the central point for measuring and
analyzing aggregate economic performance, ensuring consistent
accounting across all sectors and markets.
"""

from .economy import Economy
