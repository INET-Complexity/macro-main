from abc import ABC, abstractmethod


class PolicyRate(ABC):
    @abstractmethod
    def compute_rate(
        self,
        prev_rate: float,
        inflation: float,
        growth: float,
        central_bank_states: dict[str, float],
    ) -> float:
        pass


class ConstantPolicyRate(PolicyRate):
    def compute_rate(
        self,
        prev_rate: float,
        inflation: float,
        growth: float,
        central_bank_states: dict[str, float],
    ) -> float:
        return prev_rate


class PolednaPolicyRate(PolicyRate):
    def compute_rate(
        self,
        prev_rate: float,
        inflation: float,
        growth: float,
        central_bank_states: dict[str, float],
    ) -> float:
        return max(
            0.0,
            central_bank_states["rho"] * prev_rate
            + (1 - central_bank_states["rho"])
            * (
                central_bank_states["r_star"]
                + central_bank_states["targeted_inflation_rate"]
                + central_bank_states["xi_pi"] * (inflation - central_bank_states["targeted_inflation_rate"])
                + central_bank_states["xi_gamma"] * growth
            ),
        )
