"""Experiment orchestration for conformal prediction under perturbations."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from conformal import ConformalClassifier
from data import DatasetBundle
from models import get_model_builders

TARGET_COVERAGE = 0.9



def _flip_labels(y: np.ndarray, flip_prob: float, rng: np.random.Generator) -> np.ndarray:
    y_new = y.copy()
    classes = np.unique(y)
    n_classes = len(classes)
    flip_mask = rng.random(len(y_new)) < flip_prob

    for i in np.where(flip_mask)[0]:
        current = y_new[i]
        alternatives = classes[classes != current]
        y_new[i] = rng.choice(alternatives if n_classes > 2 else alternatives)
    return y_new



def _add_feature_noise(X: np.ndarray, noise_level: float, rng: np.random.Generator) -> np.ndarray:
    if noise_level == 0.0:
        return X
    return X + rng.normal(loc=0.0, scale=noise_level, size=X.shape)



def _subsample_train(X: np.ndarray, y: np.ndarray, n_samples: int | str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if n_samples == "full" or n_samples >= len(y):
        return X, y
    idx = rng.choice(len(y), size=n_samples, replace=False)
    return X[idx], y[idx]



def _evaluate_single_configuration(
    bundle: DatasetBundle,
    model_name: str,
    seed: int,
    alpha: float,
    distribution_noise: float = 0.0,
    label_noise: float = 0.0,
    sample_size: int | str = "full",
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    model_builders = get_model_builders()
    model = model_builders[model_name](seed)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=0.2,
        random_state=seed,
        stratify=bundle.y,
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,  # 0.25 of 0.8 => 0.2 full data calibration.
        random_state=seed,
        stratify=y_train_full,
    )

    X_train, y_train = _subsample_train(X_train, y_train, sample_size, rng)
    y_train_noisy = _flip_labels(y_train, label_noise, rng) if label_noise > 0 else y_train

    model.fit(X_train, y_train_noisy)

    conformal = ConformalClassifier(alpha=alpha).fit(model, X_calib, y_calib)
    X_test_shifted = _add_feature_noise(X_test, distribution_noise, rng)
    metrics = conformal.evaluate(X_test_shifted, y_test)

    metrics.update(
        {
            "dataset": bundle.name,
            "model": model_name,
            "seed": seed,
            "alpha": alpha,
            "target_coverage": 1 - alpha,
            "distribution_noise": distribution_noise,
            "label_noise": label_noise,
            "sample_size": sample_size,
            "coverage_gap": metrics["coverage"] - (1 - alpha),
        }
    )
    return metrics



def run_all_experiments(
    datasets: Dict[str, DatasetBundle],
    seeds: Iterable[int],
    alpha: float = 0.1,
    distribution_noise_levels: List[float] | None = None,
    label_noise_levels: List[float] | None = None,
    sample_sizes: List[int | str] | None = None,
) -> pd.DataFrame:
    distribution_noise_levels = distribution_noise_levels or [0.0, 0.1, 0.2, 0.5, 1.0]
    label_noise_levels = label_noise_levels or [0.0, 0.1, 0.2, 0.3]
    sample_sizes = sample_sizes or [100, 500, 1000, "full"]

    rows: list[Dict[str, float]] = []
    for dataset in datasets.values():
        for model_name in get_model_builders().keys():
            for seed in seeds:
                for noise in distribution_noise_levels:
                    rows.append(
                        _evaluate_single_configuration(
                            bundle=dataset,
                            model_name=model_name,
                            seed=seed,
                            alpha=alpha,
                            distribution_noise=noise,
                            label_noise=0.0,
                            sample_size="full",
                        )
                    )
                for lbl_noise in label_noise_levels:
                    rows.append(
                        _evaluate_single_configuration(
                            bundle=dataset,
                            model_name=model_name,
                            seed=seed,
                            alpha=alpha,
                            distribution_noise=0.0,
                            label_noise=lbl_noise,
                            sample_size="full",
                        )
                    )
                for sample_size in sample_sizes:
                    rows.append(
                        _evaluate_single_configuration(
                            bundle=dataset,
                            model_name=model_name,
                            seed=seed,
                            alpha=alpha,
                            distribution_noise=0.0,
                            label_noise=0.0,
                            sample_size=sample_size,
                        )
                    )

    df = pd.DataFrame(rows)

    # Create experiment type tags for downstream plotting/table logic.
    df["experiment"] = "custom"
    df.loc[(df["label_noise"] == 0.0) & (df["sample_size"].astype(str) == "full"), "experiment"] = "distribution_shift"
    df.loc[(df["distribution_noise"] == 0.0) & (df["sample_size"].astype(str) == "full"), "experiment"] = "label_noise"
    df.loc[(df["distribution_noise"] == 0.0) & (df["label_noise"] == 0.0), "experiment"] = "sample_size"

    return df



def aggregate_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "model",
        "distribution_noise",
        "label_noise",
        "sample_size",
    ]
    agg = (
        raw_df.groupby(group_cols, dropna=False)
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            avg_set_size_mean=("avg_set_size", "mean"),
            accuracy_mean=("accuracy", "mean"),
            coverage_gap_mean=("coverage_gap", "mean"),
        )
        .reset_index()
    )
    return agg
