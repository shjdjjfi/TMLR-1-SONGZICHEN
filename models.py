"""Model factory functions for classical classification baselines."""

from __future__ import annotations

from typing import Dict

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression



def get_model_builders() -> Dict[str, object]:
    """Return model builders keyed by model name."""

    return {
        "logistic_regression": lambda seed: LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
        ),
        "random_forest": lambda seed: RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "gradient_boosting": lambda seed: GradientBoostingClassifier(
            random_state=seed,
        ),
    }
