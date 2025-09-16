import warnings
from typing import Dict

import numpy as np
import torch
# Import necessary metrics from sklearn for evaluation
from sklearn.metrics import (f1_score, matthews_corrcoef,
                             multilabel_confusion_matrix, precision_score,
                             recall_score)

warnings.filterwarnings("ignore")


class ClassificationBenchmark:
    def __init__(self, dataloader, model):
        self.target = []
        self.pred = []
        self.pred_proba = []
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                logits = model(X_batch)

                self.target.extend(y_batch.detach().cpu().numpy())
                self.pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
                self.pred_proba.extend(logits.detach().cpu().numpy())

        self.target = np.array(self.target)
        self.pred = np.array(self.pred)
        self.pred_proba = np.array(self.pred_proba)

    def accuracy(self):
        # Compute the accuracy: proportion of correct predictions.
        return np.mean(self.target == self.pred)

    def precision(self, average="macro"):
        # Compute precision: TP / (TP + FP), averaged across classes.
        return precision_score(self.target, self.pred, average=average)

    def recall(self, average="macro"):
        # Compute recall: TP / (TP + FN), averaged across classes.
        return recall_score(self.target, self.pred, average=average)

    def F1(self, average="macro"):
        # Compute F1 score: harmonic mean of precision and recall, averaged across classes.
        return f1_score(self.target, self.pred, average=average)

    def informedness(self):
        # Compute informedness (Youden's J statistic): mean of (recall + specificity) - 1.
        mc_matrix = multilabel_confusion_matrix(self.target, self.pred)

        recall = mc_matrix[:, 1, 1] / (mc_matrix[:, 1, 1] + mc_matrix[:, 1, 0])
        specificity = mc_matrix[:, 0, 0] / (mc_matrix[:, 0, 0] + mc_matrix[:, 0, 1])

        return np.mean(recall + specificity) - 1

    def markedness(self):
        # Compute markedness: mean of (precision + NPV) - 1.
        # NPV is Negative Predictive Value.
        mc_matrix = multilabel_confusion_matrix(self.target, self.pred)

        precision = mc_matrix[:, 1, 1] / (mc_matrix[:, 1, 1] + mc_matrix[:, 0, 1])
        npv = mc_matrix[:, 0, 0] / (mc_matrix[:, 0, 0] + mc_matrix[:, 1, 0])

        return np.mean(precision + npv) - 1

    def matthews(self):
        # Compute Matthews Correlation Coefficient (MCC).
        return matthews_corrcoef(self.target, self.pred)


def evaluate_classification(dataloader, model) -> Dict[str, float]:
    # Evaluate a classification model using the ClassificationBenchmark class.
    bench = ClassificationBenchmark(dataloader, model)

    return {
        "accuracy": bench.accuracy(),
        "precision": bench.precision(),
        "recall": bench.recall(),
        "F1": bench.F1(),
        "informedness": bench.informedness(),
        "markedness": bench.markedness(),
        "matthews_corrcoef": bench.matthews(),
    }
