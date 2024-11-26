from abc import ABC, abstractmethod

import numpy as np


class ProductionSetter(ABC):
    def compute_production(
        self,
        desired_production: np.ndarray,
        current_labour_inputs: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
    ) -> np.ndarray:
        limiting_stock = self.compute_limiting_stock(
            current_limiting_intermediate_inputs,
            current_limiting_capital_inputs,
        )
        return np.amin([desired_production, current_labour_inputs, limiting_stock], axis=0)

    @abstractmethod
    def compute_limiting_intermediate_inputs_stock(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_limiting_capital_inputs_stock(
        self,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        capital_inputs_utilisation_rate: float,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        pass

    @staticmethod
    def compute_limiting_stock(
        limiting_intermediate_inputs_stock: np.ndarray,
        limiting_capital_inputs_stock: np.ndarray,
    ) -> np.ndarray:
        return np.amin(
            [limiting_intermediate_inputs_stock, limiting_capital_inputs_stock],
            axis=0,
        )

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
    def compute_limiting_intermediate_inputs_stock(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        return np.multiply(
            intermediate_inputs_productivity_matrix,
            intermediate_inputs_stock,
            out=np.full(intermediate_inputs_productivity_matrix.shape, np.inf),
            where=intermediate_inputs_productivity_matrix != np.inf,
        ).min(axis=1)

    def compute_limiting_capital_inputs_stock(
        self,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        capital_inputs_utilisation_rate: float,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        return np.multiply(
            capital_inputs_productivity_matrix,
            capital_inputs_stock,
            out=np.full(capital_inputs_productivity_matrix.shape, np.inf),
            where=capital_inputs_productivity_matrix != np.inf,
        ).min(axis=1)

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
    def compute_limiting_intermediate_inputs_stock(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        rescaled_intermediate_inputs = np.multiply(
            intermediate_inputs_productivity_matrix,
            intermediate_inputs_stock,
            out=np.full(intermediate_inputs_productivity_matrix.shape, np.inf),
            where=intermediate_inputs_productivity_matrix != np.inf,
        )
        rescaled_intermediate_inputs[goods_criticality_matrix == 0.0] = np.inf
        return rescaled_intermediate_inputs.min(axis=1)

    def compute_limiting_capital_inputs_stock(
        self,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        capital_inputs_utilisation_rate: float,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        rescaled_capital_inputs = np.multiply(
            capital_inputs_productivity_matrix,
            capital_inputs_stock,
            out=np.full(capital_inputs_productivity_matrix.shape, np.inf),
            where=capital_inputs_productivity_matrix != np.inf,
        )
        rescaled_capital_inputs[goods_criticality_matrix == 0.0] = np.inf
        return rescaled_capital_inputs.min(axis=1)

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
    def compute_limiting_intermediate_inputs_stock(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        rescaled_intermediate_inputs = np.multiply(
            intermediate_inputs_productivity_matrix,
            intermediate_inputs_stock,
            out=np.full(intermediate_inputs_productivity_matrix.shape, np.inf),
            where=intermediate_inputs_productivity_matrix != np.inf,
        )
        rescaled_intermediate_inputs[goods_criticality_matrix < 1.0] = np.inf
        return rescaled_intermediate_inputs.min(axis=1)

    def compute_limiting_capital_inputs_stock(
        self,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        capital_inputs_utilisation_rate: float,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        rescaled_capital_inputs = np.multiply(
            capital_inputs_productivity_matrix,
            capital_inputs_stock,
            out=np.full(capital_inputs_productivity_matrix.shape, np.inf),
            where=capital_inputs_productivity_matrix != np.inf,
        )
        rescaled_capital_inputs[goods_criticality_matrix < 1.0] = np.inf
        return rescaled_capital_inputs.min(axis=1)

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
    def compute_limiting_intermediate_inputs_stock(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        return np.multiply(
            intermediate_inputs_productivity_matrix,
            intermediate_inputs_stock,
            out=np.zeros_like(intermediate_inputs_productivity_matrix),
            where=intermediate_inputs_productivity_matrix != np.inf,
        ).sum(axis=1)

    def compute_limiting_capital_inputs_stock(
        self,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        capital_inputs_utilisation_rate: float,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        return np.multiply(
            capital_inputs_productivity_matrix,
            capital_inputs_stock,
            out=np.zeros_like(capital_inputs_productivity_matrix),
            where=capital_inputs_productivity_matrix != np.inf,
        ).sum(axis=1)

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
        )
        used_intermediate_inputs = (
            total_used_intermediate_inputs.sum(axis=1)[:, None]
            * intermediate_inputs_stock
            / intermediate_inputs_stock.sum(axis=1, keepdims=True)
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
        used_capital_inputs = (
            used_capital_inputs.sum(axis=1)[:, None]
            * capital_inputs_stock
            / capital_inputs_stock.sum(axis=1, keepdims=True)
        )
        return used_capital_inputs


class UnconstrainedProduction(ProductionSetter):
    def compute_limiting_intermediate_inputs_stock(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        return np.multiply(
            intermediate_inputs_productivity_matrix,
            intermediate_inputs_stock,
            out=np.full(intermediate_inputs_productivity_matrix.shape, np.inf),
            where=intermediate_inputs_productivity_matrix != np.inf,
        ).min(axis=1)

    def compute_limiting_capital_inputs_stock(
        self,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        capital_inputs_utilisation_rate: float,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        return np.multiply(
            capital_inputs_productivity_matrix,
            capital_inputs_stock,
            out=np.full(capital_inputs_productivity_matrix.shape, np.inf),
            where=capital_inputs_productivity_matrix != np.inf,
        ).min(axis=1)

    def compute_intermediate_inputs_used(
        self,
        realised_production: np.ndarray,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        return np.zeros(intermediate_inputs_stock.shape)

    def compute_capital_inputs_used(
        self,
        realised_production: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
    ) -> np.ndarray:
        return np.zeros(capital_inputs_stock.shape)

    def compute_production(
        self,
        desired_production: np.ndarray,
        current_labour_inputs: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
    ) -> np.ndarray:
        return desired_production
