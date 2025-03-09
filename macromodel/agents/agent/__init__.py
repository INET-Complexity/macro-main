"""Base agent module.

This module provides the foundational Agent class that defines common
functionality shared by all economic agents in the model. It includes:

1. Base Agent Interface:
   - Common attributes and methods
   - State management
   - Time series tracking
   - Market interaction protocols

2. Agent Utilities:
   - Helper functions
   - Common calculations
   - Shared behaviors

All specific agent types (households, firms, etc.) inherit from and
extend this base functionality.
"""

from .agent import Agent
