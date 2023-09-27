import numpy as np

from abc import abstractmethod, ABC


class DemandEstimator(ABC):
    @abstractmethod
    def compute_estimated_demand(
        self,
        previous_demand: np.ndarray,
        estimated_sectoral_growth: np.ndarray,
        estimated_growth_by_firm: np.ndarray,
        firm_industry: np.ndarray,
    ) -> np.ndarray:
        pass


class DefaultDemandEstimator(DemandEstimator):
    def compute_estimated_demand(
        self,
        previous_demand: np.ndarray,
        estimated_sectoral_growth: np.ndarray,
        estimated_growth_by_firm: np.ndarray,
        firm_industry: np.ndarray,
    ) -> np.ndarray:
        return (1 + estimated_sectoral_growth[firm_industry]) * (1 + estimated_growth_by_firm) * previous_demand
