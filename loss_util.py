import numpy as np

def mean_squared_error(act: np.ndarray, pred: np.ndarray) -> float:
    diff = pred - act
    differences_squared = diff ** 2
    mean_diff = differences_squared.mean()
    
    return mean_diff