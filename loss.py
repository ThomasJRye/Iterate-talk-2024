import numpy as np
import argparse


def mean_squared_error(act: np.ndarray, pred: np.ndarray) -> float:
    diff = pred - act
    print("difference", diff)
    differences_squared = diff ** 2
    print("difference suqared", differences_squared)
    mean_diff = differences_squared.mean()
    
    return mean_diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the loss function.")
    parser.add_argument("y_true", type=float, help="True value (0 or 1)")
    parser.add_argument("y_pred", type=float, help="Predicted value (between 0 and 1)")

    args = parser.parse_args()

    y_true = args.y_true
    y_pred = args.y_pred

    result = mean_squared_error(np.array([y_true]), np.array([y_pred]))
    print(f"Loss: {result}")



