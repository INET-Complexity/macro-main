import numpy as np

#Ninety ten ratio
def compute_ninetyten(data: np.ndarray) -> np.ndarray:
    low_ten = np.percentile(data, 10, axis=1)
    high_ten = np.percentile(data, 90, axis=1)
    return high_ten / low_ten

#https://github.com/oliviaguest
def compute_gini(data: np.ndarray) -> np.ndarray:
    def flat_gini(data: np.ndarray) -> np.ndarray:
        sorted_data = np.sort(data)
        index = np.arange(1,sorted_data.shape[0]+1) #index per array element
        n = sorted_data.shape[0] #number of array elements
        return ((np.sum((2 * index - n  - 1) * sorted_data)) / (n * np.sum(sorted_data))) #Gini coefficient
    return np.apply_along_axis(flat_gini, axis=1, arr=data)

# Example usage:
#data = np.array([[1, 3, 2], [4, 6, 5], [9, 8, 7], [1970, 11, 12]])
#print(data.shape)
#result = compute_gini(data)
#print(result)