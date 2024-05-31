import numpy as np

from abc import abstractmethod, ABC


class ExcessDemandSetter(ABC):
    @abstractmethod
    def set_maximum_excess_demand(
        self,
        n_exporters: int,
    ) -> np.ndarray:
        pass


class ZeroExcessDemandSetter(ExcessDemandSetter):
    def set_maximum_excess_demand(
        self,
        n_exporters: int,
    ) -> np.ndarray:
        return np.zeros(n_exporters)


class InfinityExcessDemandSetter(ExcessDemandSetter):
    def set_maximum_excess_demand(
        self,
        n_exporters: int,
    ) -> np.ndarray:
        return np.full(n_exporters, np.inf)
