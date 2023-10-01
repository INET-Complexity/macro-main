from abc import abstractmethod, ABC


class PolicyRate(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute_rate(self, prev_rate: float) -> float:
        pass


class ConstantPolicyRate(PolicyRate):
    def compute_rate(self, prev_rate: float) -> float:
        return prev_rate
