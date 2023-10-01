import numpy as np

from abc import abstractmethod, ABC


class SentimentSetter(ABC):
    @abstractmethod
    def compute_sentiment(self, n_industries: int) -> np.ndarray:
        pass


class ConstantSentimentSetter(SentimentSetter):
    def __init__(self, value: float):
        self.sentiment_value = value

    def compute_sentiment(self, n_industries: np.ndarray) -> np.ndarray:
        return np.full(n_industries, self.sentiment_value)
