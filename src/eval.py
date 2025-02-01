# from sklearn.metrics import f1_score
# import numpy as np

# def compute_metrics(eval_pred):
#     """
#     Computes F1 score for evaluation.
#     """
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
    
#     # Ensure labels and predictions are in the same format
#     if len(labels.shape) > 1 and labels.shape[1] > 1:
#         labels = np.argmax(labels, axis=1)
#     return {"f1": f1_score(labels, predictions, average="weighted")}

import evaluate
import numpy as np

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def sigmoid(x):
   return 1/(1 + np.exp(-x))

def compute_metrics(eval_pred):

   predictions, labels = eval_pred
   predictions = sigmoid(predictions)
   predictions = (predictions > 0.5).astype(int).reshape(-1)
   return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))
# references=labels.astype(int).reshape(-1))
