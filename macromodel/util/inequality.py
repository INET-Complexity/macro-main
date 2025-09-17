import numpy as np


class NinetyRatios:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.ninety = np.percentile(data, 90, axis=1)

    def compute_ninetyten(self) -> np.ndarray:
        ten = np.percentile(self.data, 10, axis=1)
        return self.ninety / ten

    def compute_ninetyfifty(self) -> np.ndarray:
        fifty = np.percentile(self.data, 50, axis=1)
        return self.ninety / fifty

    def compute_palma(self) -> np.ndarray:
        forty = np.percentile(self.data, 40, axis=1)
        return self.ninety / forty


# https://github.com/oliviaguest
def compute_gini(data: np.ndarray) -> np.ndarray:
    def flat_gini(data: np.ndarray) -> np.ndarray:
        sorted_data = np.sort(data)
        index = np.arange(1, sorted_data.shape[0] + 1)  # index per array element
        n = sorted_data.shape[0]  # number of array elements
        return (np.sum((2 * index - n - 1) * sorted_data)) / (n * np.sum(sorted_data))  # Gini coefficient

    return np.apply_along_axis(flat_gini, axis=1, arr=data)


# Example usage:
# data = np.array([[1, 3, 2], [4, 6, 5], [9, 8, 7], [1970, 11, 12]])
# print(data.shape)
# result = compute_gini(data)
# print(result)
