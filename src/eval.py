from sklearn.metrics import f1_score
import numpy as np

def compute_metrics(eval_pred):
    """
    Computes F1 score for evaluation.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Ensure labels and predictions are in the same format
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)
    return {"f1": f1_score(labels, predictions, average="weighted")}
