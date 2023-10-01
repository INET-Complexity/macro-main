import numpy as np

from abc import abstractmethod, ABC


class ProductionSetter(ABC):
    def __init__(self, production_noise_std: float = 0.0):
        self.production_noise_std = production_noise_std

    @abstractmethod
    def compute_limiting_stock(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float,
        capital_inputs_utilisation_rate: float,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_production(
        self,
        desired_production: np.ndarray,
        current_labour_inputs: np.ndarray,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float,
        capital_inputs_utilisation_rate: float,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_intermediate_inputs_used(
        self,
        realised_production: np.ndarray,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_capital_inputs_used(
        self,
        realised_production: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        pass


class PureLeontief(ProductionSetter):
    def compute_limiting_stock(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float,
        capital_inputs_utilisation_rate: float,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        rescaled_intermediate_inputs = intermediate_inputs_utilisation_rate * np.multiply(
            intermediate_inputs_productivity_matrix,
            intermediate_inputs_stock,
            out=np.full(capital_inputs_productivity_matrix.shape, np.inf),
            where=intermediate_inputs_productivity_matrix != np.inf,
        )
        rescaled_capital_inputs = capital_inputs_utilisation_rate * np.multiply(
            capital_inputs_productivity_matrix,
            capital_inputs_stock,
            out=np.full(capital_inputs_productivity_matrix.shape, np.inf),
            where=capital_inputs_productivity_matrix != np.inf,
        )
        return np.amin(
            [
                rescaled_intermediate_inputs.min(axis=1),
                rescaled_capital_inputs.min(axis=1),
            ],
            axis=0,
        )

    def compute_production(
        self,
        desired_production: np.ndarray,
        current_labour_inputs: np.ndarray,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float,
        capital_inputs_utilisation_rate: float,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        limiting_stock = self.compute_limiting_stock(
            intermediate_inputs_productivity_matrix,
            intermediate_inputs_stock,
            capital_inputs_productivity_matrix,
            capital_inputs_stock,
            intermediate_inputs_utilisation_rate,
            capital_inputs_utilisation_rate,
            goods_criticality_matrix,
        )

        # Leontief in the main inputs
        noise = np.random.normal(
            0,
            self.production_noise_std,
            size=desired_production.shape,
        )
        return (1 + noise) * np.amin(
            [
                desired_production,
                current_labour_inputs,
                limiting_stock,
            ],
            axis=0,
        )

    def compute_intermediate_inputs_used(
        self,
        realised_production: np.ndarray,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        return np.divide(
            realised_production[:, None],
            intermediate_inputs_productivity_matrix,
            out=np.zeros_like(intermediate_inputs_productivity_matrix),
            where=intermediate_inputs_productivity_matrix != 0.0,
        )

    def compute_capital_inputs_used(
        self,
        realised_production: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        used_capital_inputs = realised_production[:, None] * capital_inputs_depreciation_matrix
        used_capital_inputs[used_capital_inputs == np.inf] = 0.0
        used_capital_inputs[used_capital_inputs == -np.inf] = 0.0
        return used_capital_inputs


class CriticalAndImportantLeontief(ProductionSetter):
    def compute_limiting_stock(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float,
        capital_inputs_utilisation_rate: float,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        rescaled_intermediate_inputs = intermediate_inputs_utilisation_rate * np.multiply(
            intermediate_inputs_productivity_matrix,
            intermediate_inputs_stock,
            out=np.full(capital_inputs_productivity_matrix.shape, np.inf),
            where=intermediate_inputs_productivity_matrix != np.inf,
        )
        rescaled_intermediate_inputs[goods_criticality_matrix == 0.0] = np.inf
        rescaled_capital_inputs = capital_inputs_utilisation_rate * np.multiply(
            capital_inputs_productivity_matrix,
            capital_inputs_stock,
            out=np.full(capital_inputs_productivity_matrix.shape, np.inf),
            where=capital_inputs_productivity_matrix != np.inf,
        )
        rescaled_capital_inputs[goods_criticality_matrix == 0.0] = np.inf

        return np.amin(
            [
                rescaled_intermediate_inputs.min(axis=1),
                rescaled_capital_inputs.min(axis=1),
            ],
            axis=0,
        )

    def compute_production(
        self,
        desired_production: np.ndarray,
        current_labour_inputs: np.ndarray,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float,
        capital_inputs_utilisation_rate: float,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        limiting_stock = self.compute_limiting_stock(
            intermediate_inputs_productivity_matrix,
            intermediate_inputs_stock,
            capital_inputs_productivity_matrix,
            capital_inputs_stock,
            intermediate_inputs_utilisation_rate,
            capital_inputs_utilisation_rate,
            goods_criticality_matrix,
        )

        # Leontief in the main inputs
        noise = np.random.normal(
            0,
            self.production_noise_std,
            size=desired_production.shape,
        )
        return (1 + noise) * np.amin(
            [desired_production, current_labour_inputs, limiting_stock],
            axis=0,
        )

    def compute_intermediate_inputs_used(
        self,
        realised_production: np.ndarray,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        used_intermediate_inputs = np.divide(
            realised_production[:, None],
            intermediate_inputs_productivity_matrix,
            out=np.zeros_like(intermediate_inputs_productivity_matrix),
            where=intermediate_inputs_productivity_matrix != 0.0,
        )
        used_intermediate_inputs[goods_criticality_matrix == 0.0] = 0.0
        return used_intermediate_inputs

    def compute_capital_inputs_used(
        self,
        realised_production: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        used_capital_inputs = realised_production[:, None] * capital_inputs_depreciation_matrix
        used_capital_inputs[used_capital_inputs == np.inf] = 0.0
        used_capital_inputs[used_capital_inputs == -np.inf] = 0.0
        used_capital_inputs[goods_criticality_matrix == 0.0] = 0.0
        return used_capital_inputs


class CriticalLeontief(ProductionSetter):
    def compute_limiting_stock(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float,
        capital_inputs_utilisation_rate: float,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        rescaled_intermediate_inputs = intermediate_inputs_utilisation_rate * np.multiply(
            intermediate_inputs_productivity_matrix,
            intermediate_inputs_stock,
            out=np.full(capital_inputs_productivity_matrix.shape, np.inf),
            where=intermediate_inputs_productivity_matrix != np.inf,
        )
        rescaled_intermediate_inputs[goods_criticality_matrix < 1.0] = np.inf
        rescaled_capital_inputs = capital_inputs_utilisation_rate * np.multiply(
            capital_inputs_productivity_matrix,
            capital_inputs_stock,
            out=np.full(capital_inputs_productivity_matrix.shape, np.inf),
            where=capital_inputs_productivity_matrix != np.inf,
        )
        rescaled_capital_inputs[goods_criticality_matrix < 1.0] = np.inf

        return np.amin(
            [
                rescaled_intermediate_inputs.min(axis=1),
                rescaled_capital_inputs.min(axis=1),
            ],
            axis=0,
        )

    def compute_production(
        self,
        desired_production: np.ndarray,
        current_labour_inputs: np.ndarray,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float,
        capital_inputs_utilisation_rate: float,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        limiting_stock = self.compute_limiting_stock(
            intermediate_inputs_productivity_matrix,
            intermediate_inputs_stock,
            capital_inputs_productivity_matrix,
            capital_inputs_stock,
            intermediate_inputs_utilisation_rate,
            capital_inputs_utilisation_rate,
            goods_criticality_matrix,
        )

        # Leontief in the main inputs
        noise = np.random.normal(
            0,
            self.production_noise_std,
            size=desired_production.shape,
        )
        return (1 + noise) * np.amin(
            [desired_production, current_labour_inputs, limiting_stock],
            axis=0,
        )

    def compute_intermediate_inputs_used(
        self,
        realised_production: np.ndarray,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        used_intermediate_inputs = np.divide(
            realised_production[:, None],
            intermediate_inputs_productivity_matrix,
            out=np.zeros_like(intermediate_inputs_productivity_matrix),
            where=intermediate_inputs_productivity_matrix != 0.0,
        )
        used_intermediate_inputs[goods_criticality_matrix < 1.0] = 0.0
        return used_intermediate_inputs

    def compute_capital_inputs_used(
        self,
        realised_production: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        used_capital_inputs = realised_production[:, None] * capital_inputs_depreciation_matrix
        used_capital_inputs[used_capital_inputs == np.inf] = 0.0
        used_capital_inputs[used_capital_inputs == -np.inf] = 0.0
        used_capital_inputs[goods_criticality_matrix < 1.0] = 0.0
        return used_capital_inputs


class Linear(ProductionSetter):
    def compute_limiting_stock(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float,
        capital_inputs_utilisation_rate: float,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        rescaled_intermediate_inputs = intermediate_inputs_utilisation_rate * np.multiply(
            intermediate_inputs_productivity_matrix,
            intermediate_inputs_stock,
            out=np.zeros_like(capital_inputs_productivity_matrix),
            where=intermediate_inputs_productivity_matrix != np.inf,
        )
        rescaled_capital_inputs = capital_inputs_utilisation_rate * np.multiply(
            capital_inputs_productivity_matrix,
            capital_inputs_stock,
            out=np.zeros_like(capital_inputs_productivity_matrix),
            where=capital_inputs_productivity_matrix != np.inf,
        )

        return np.amin(
            [
                rescaled_intermediate_inputs.sum(axis=1),
                rescaled_capital_inputs.sum(axis=1),
            ],
            axis=0,
        )

    def compute_production(
        self,
        desired_production: np.ndarray,
        current_labour_inputs: np.ndarray,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float,
        capital_inputs_utilisation_rate: float,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        limiting_stock = self.compute_limiting_stock(
            intermediate_inputs_productivity_matrix,
            intermediate_inputs_stock,
            capital_inputs_productivity_matrix,
            capital_inputs_stock,
            intermediate_inputs_utilisation_rate,
            capital_inputs_utilisation_rate,
            goods_criticality_matrix,
        )

        # Leontief in the main inputs
        noise = np.random.normal(
            0,
            self.production_noise_std,
            size=desired_production.shape,
        )
        return (1 + noise) * np.amin(
            [desired_production, current_labour_inputs, limiting_stock],
            axis=0,
        )

    def compute_intermediate_inputs_used(
        self,
        realised_production: np.ndarray,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        total_used_intermediate_inputs = np.divide(
            realised_production[:, None],
            intermediate_inputs_productivity_matrix,
            out=np.zeros_like(intermediate_inputs_productivity_matrix),
            where=intermediate_inputs_productivity_matrix != 0.0,
        ).sum(axis=1)
        used_intermediate_inputs = (
            total_used_intermediate_inputs[:, None] * intermediate_inputs_stock / intermediate_inputs_stock.sum(axis=1)
        )

        return used_intermediate_inputs

    def compute_capital_inputs_used(
        self,
        realised_production: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        used_capital_inputs = realised_production[:, None] * capital_inputs_depreciation_matrix
        used_capital_inputs[used_capital_inputs == np.inf] = 0.0
        used_capital_inputs[used_capital_inputs == -np.inf] = 0.0
        total_used_capital_inputs = used_capital_inputs.sum(axis=1)
        used_capital_inputs = (
            total_used_capital_inputs[:, None] * capital_inputs_stock / capital_inputs_stock.sum(axis=1)
        )

        return used_capital_inputs
