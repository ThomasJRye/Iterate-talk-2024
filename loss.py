import numpy as np
import argparse

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss(y_true, y_pred):
    if y_true == y_pred:
        return 0
        # Ensure y_pred is within the range (0, 1)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return abs(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def loss_rounded(y_true, y_pred):
    return round(loss(y_true, y_pred), 3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the loss function.")
    parser.add_argument("y_true", type=float, help="True value (0 or 1)")
    parser.add_argument("y_pred", type=float, help="Predicted value (between 0 and 1)")

    args = parser.parse_args()

    y_true = args.y_true
    y_pred = args.y_pred

    result = loss_rounded(y_true, y_pred)
    print(f"Loss: {result}")



