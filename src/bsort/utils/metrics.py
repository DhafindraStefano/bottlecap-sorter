
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def calculate_metrics(y_true, y_pred):
    """
    Calculates classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and f1 score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def get_confusion_matrix(y_true, y_pred):
    """
    Returns the confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        np.ndarray: Confusion matrix.
    """
    return confusion_matrix(y_true, y_pred)
