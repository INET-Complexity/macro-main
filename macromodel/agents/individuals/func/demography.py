"""Individual demographic dynamics management.

This module implements strategies for managing individual demographic changes
through:
- Population size updates
- Birth and death events
- Workforce entry/exit
- Age-based transitions

The implementation handles:
- Natural population changes
- Labor force dynamics
- Demographic transitions
- Age structure evolution
"""

from abc import ABC, abstractmethod


class IndividualDemography(ABC):
    """Abstract base class for individual demographic management.

    This class defines strategies for handling demographic changes in
    the individual population through:
    - Population size tracking
    - Life events (birth/death)
    - Labor force participation changes
    - Age-based transitions

    The strategies consider:
    - Natural population dynamics
    - Workforce demographics
    - Age structure changes
    - Population stability
    """

    @abstractmethod
    def update(
        self,
        prev_n_individuals: float,
    ) -> float:
        """Update total population size.

        Args:
            prev_n_individuals (float): Previous period's population

        Returns:
            float: New population size
        """
        pass

    @abstractmethod
    def check_for_death(
        self,
    ) -> None:
        """Process individual death events."""
        pass

    @abstractmethod
    def check_for_birth(
        self,
    ) -> None:
        """Process individual birth events."""
        pass

    @abstractmethod
    def individuals_joining_the_workforce(
        self,
    ) -> None:
        """Process individuals entering the labor force."""
        pass

    @abstractmethod
    def individuals_leaving_the_workforce(
        self,
    ) -> None:
        """Process individuals exiting the labor force."""
        pass


class NoAging(IndividualDemography):
    """Static demographic implementation with no population changes.

    This class implements a simplified approach that:
    - Maintains constant population size
    - Ignores demographic transitions
    - Preserves workforce composition
    - Keeps age structure static

    Used for:
    - Model testing
    - Baseline scenarios
    - Short-term simulations
    - Controlled experiments
    """

    def update(
        self,
        prev_n_individuals: float,
    ) -> float:
        """Maintain constant population size.

        Args:
            prev_n_individuals (float): Previous period's population

        Returns:
            float: Same population size (no change)
        """
        return prev_n_individuals

    def check_for_death(
        self,
    ) -> None:
        """No death events processed."""
        pass

    def check_for_birth(
        self,
    ) -> None:
        """No birth events processed."""
        pass

    def individuals_joining_the_workforce(
        self,
    ) -> None:
        """No workforce entry events processed."""
        pass

    def individuals_leaving_the_workforce(
        self,
    ) -> None:
        """No workforce exit events processed."""
        pass
