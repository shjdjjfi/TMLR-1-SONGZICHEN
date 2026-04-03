"""Manual split conformal prediction for classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ConformalClassifier:
    """Split conformal predictor using model probabilities and score 1 - p_y."""

    alpha: float = 0.1

    def fit(self, model, X_calib: np.ndarray, y_calib: np.ndarray) -> "ConformalClassifier":
        self.model = model
        self.classes_ = model.classes_

        prob_calib = model.predict_proba(X_calib)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        calib_idx = np.array([class_to_idx[y] for y in y_calib])

        # Nonconformity scores for true labels.
        self.calib_scores_ = 1.0 - prob_calib[np.arange(len(y_calib)), calib_idx]
        n = len(self.calib_scores_)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.qhat_ = float(np.quantile(self.calib_scores_, q_level, method="higher"))
        return self

    def predict_sets(self, X: np.ndarray) -> np.ndarray:
        probs = self.model.predict_proba(X)
        scores = 1.0 - probs
        return scores <= self.qhat_

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        pred_sets = self.predict_sets(X)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[y] for y in y_true])

        covered = pred_sets[np.arange(len(y_true)), y_idx]
        set_sizes = pred_sets.sum(axis=1)
        accuracy = np.mean(self.model.predict(X) == y_true)

        return {
            "coverage": float(np.mean(covered)),
            "avg_set_size": float(np.mean(set_sizes)),
            "accuracy": float(accuracy),
        }
