from abc import ABC, abstractmethod


class RoWInflationSetter(ABC):
    @abstractmethod
    def compute_inflation(self, average_country_ppi_inflation: float) -> float:
        pass


class DefaultRoWInflationSetter(RoWInflationSetter):
    def compute_inflation(self, average_country_ppi_inflation: float) -> float:
        return average_country_ppi_inflation
