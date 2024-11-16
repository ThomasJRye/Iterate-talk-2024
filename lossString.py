import numpy as np
import argparse

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
    parser.add_argument("string1", type=str, help="The first string")
    parser.add_argument("string2", type=str, help="The second string")

    args = parser.parse_args()

    y_true = args.string1
    y_pred = args.string2

    # Convert strings to their Unicode numerical values
    unicode_true = [ord(char) for char in y_true]
    unicode_pred = [ord(char) for char in y_pred]

    print(f"Unicode values for '{y_true}': {unicode_true}")
    print(f"Unicode values for '{y_pred}': {unicode_pred}")

    # normalize values to be between 0 and 1
    unicode_true = [x / 255 for x in unicode_true]
    unicode_pred = [x / 255 for x in unicode_pred]

    # Add 0s to the end of the shorter string to make them the same length
    max_len = max(len(unicode_true), len(unicode_pred))
    unicode_true += [0] * (max_len - len(unicode_true))
    unicode_pred += [0] * (max_len - len(unicode_pred))

    print("loss: " + str(loss(1, 1)))
    # For each character in the string, calculate the loss
    total_loss = 0
    for true_val, pred_val in zip(unicode_true, unicode_pred):
        print(f"True value: {true_val}, Predicted value: {pred_val}")
        total_loss += loss_rounded(true_val, pred_val)

    mean_loss = total_loss / max_len
    print(f"Mean loss: {mean_loss}")
    

