from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BoughtGoodsDistributor(ABC):
    @abstractmethod
    def distribute_bought_goods(
        self,
        desired_intermediate_inputs: np.ndarray,
        desired_investment: np.ndarray,
        buy_real: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass


class BoughtGoodsDistributorIIPrio(BoughtGoodsDistributor):
    def __init__(self):
        pass

    def distribute_bought_goods(
        self,
        desired_intermediate_inputs: np.ndarray,
        desired_investment: np.ndarray,
        buy_real: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.minimum(desired_intermediate_inputs, buy_real),
            np.maximum(0.0, buy_real - desired_intermediate_inputs),
        )


class BoughtGoodsDistributorEvenly(BoughtGoodsDistributor):
    def __init__(self):
        pass

    def distribute_bought_goods(
        self,
        desired_intermediate_inputs: np.ndarray,
        desired_investment: np.ndarray,
        buy_real: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        real_intermediate_inputs = (
            desired_intermediate_inputs / (desired_intermediate_inputs + desired_investment) * buy_real
        )
        return (
            real_intermediate_inputs,
            (buy_real - real_intermediate_inputs),
        )
