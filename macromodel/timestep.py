"""Time progression management for the macroeconomic model.

This module provides a simple but essential time-tracking mechanism for the simulation,
managing the progression of months and years. It handles the rollover from month 12
to the next year automatically.
"""


class Timestep:
    """A time-tracking class that manages simulation time progression.

    This class maintains the current year and month of the simulation, providing
    methods to advance time and represent the current timestep. It automatically
    handles year transitions when months progress past December.

    Attributes:
        year (int): Current year of the simulation
        month (int): Current month of the simulation (1-12)
    """

    def __init__(self, year: int, month: int):
        """Initialize a new timestep.

        Args:
            year (int): Starting year
            month (int): Starting month (1-12)
        """
        self.year = year
        self.month = month

    def __str__(self):
        """Convert timestep to string representation.

        Returns:
            str: Timestep in 'YYYY-MM' format
        """
        return str(self.year) + "-" + str(self.month)

    def step(self) -> None:
        """Advance the timestep by one month.

        Increments the month counter and handles year rollover when
        transitioning from December to January.
        """
        self.month += 1
        if self.month == 13:
            self.year += 1
            self.month = 1
