"""Country module.

This module implements a complete national economy with interacting agents
and markets. Key features include:

1. Economic Structure:
   - Agent populations (individuals, households, firms)
   - Market mechanisms (labor, credit, housing, goods)
   - Government institutions (fiscal and monetary)
   - International linkages

2. Dynamic Processes:
   - Market clearing mechanisms
   - Agent decision-making
   - Policy implementation
   - Economic evolution

3. System Integration:
   - Cross-market coordination
   - Policy transmission
   - Behavioral feedback
   - Aggregate outcomes

The country module serves as the primary container for simulating
a national economy, coordinating all agents and markets while
maintaining consistency in economic flows and stocks.
"""

from .country import Country
