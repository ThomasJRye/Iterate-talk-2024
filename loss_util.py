import numpy as np

def mean_squared_error(act: np.ndarray, pred: np.ndarray) -> float:
    diff = pred - act
    print("difference", diff)
    differences_squared = diff ** 2
    print("difference suqared", differences_squared)
    mean_diff = differences_squared.mean()
    
    return mean_diff