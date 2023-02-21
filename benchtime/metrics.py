import numpy as np


def smape(y_true, y_pred, **kwargs):
    return (
        1
        / len(y_true)
        * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)) * 100)
    )
