from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    jaccard_score,
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from benchtime.metrics import smape

REGRESSION_METRICS = {
    "explained_variance_score": explained_variance_score,
    "max_error": max_error,
    "mean_absolute_error": mean_absolute_error,
    "mean_squared_error": mean_squared_error,
    "root_mean_squared_error": mean_squared_error,
    "median_absolute_error": median_absolute_error,
    "r2_score": r2_score,
    "mean_absolute_percentage_error": mean_absolute_percentage_error,
    "symmetric_mean_absolute_percentage_error": smape,
}


REGRESSION_METRICS_KWARGS = {
    "explained_variance_score": dict(),
    "max_error": dict(),
    "mean_absolute_error": dict(),
    "mean_squared_error": dict(),
    "mean_squared_log_error": dict(),
    "root_mean_squared_error": {"squared": False},
    "median_absolute_error": dict(),
    "r2_score": dict(),
    "mean_poisson_deviance": dict(),
    "mean_gamma_deviance": dict(),
    "mean_absolute_percentage_error": dict(),
    "symmetric_mean_absolute_percentage_error": dict(),
    "d2_absolute_error_score": dict(),
    "d2_pinball_score": dict(),
    "d2_tweedie_score": dict(),
}


CLASSIFICATION_METRICS = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "jaccard": jaccard_score,
    "f1": f1_score,
}

CLASSIFICATION_METRICS_KWARGS = {
    "accuracy": dict(),
    "balanced_accuracy": dict(),
    "precision": {"average": "macro"},
    "recall": {"average": "macro"},
    "jaccard": {"average": "macro"},
    "f1": {"average": "macro"},
}


def score_regression(y_true, y_pred):
    scores = dict()
    for name, metric in REGRESSION_METRICS.items():
        scores[name] = metric(y_true, y_pred, **REGRESSION_METRICS_KWARGS[name])
    return scores


def score_classification(y_true, y_pred):
    scores = dict()
    for name, metric in CLASSIFICATION_METRICS.items():
        scores[name] = metric(y_true, y_pred, **CLASSIFICATION_METRICS_KWARGS[name])
    return scores


if __name__ == "__main__":
    pass
