"""Dataset loading and preprocessing utilities for conformal prediction experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


@dataclass
class DatasetBundle:
    """Container for processed dataset data and metadata."""

    name: str
    X: np.ndarray
    y: np.ndarray
    class_labels: np.ndarray



def _build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )



def _process_dataframe(name: str, X_raw: pd.DataFrame, y_raw: pd.Series) -> DatasetBundle:
    y_series = pd.Series(y_raw).astype(str)
    valid_idx = y_series.notna()
    X_raw = X_raw.loc[valid_idx].reset_index(drop=True)
    y_series = y_series.loc[valid_idx].reset_index(drop=True)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_series.values)

    preprocessor = _build_preprocessor(X_raw)
    X = preprocessor.fit_transform(X_raw)

    return DatasetBundle(name=name, X=np.asarray(X), y=np.asarray(y), class_labels=label_encoder.classes_)



def load_breast_cancer_dataset() -> DatasetBundle:
    ds = load_breast_cancer(as_frame=True)
    X = ds.data.copy()
    y = ds.target.copy().astype(str)
    return _process_dataframe("breast_cancer", X, y)



def load_wine_binary_dataset() -> DatasetBundle:
    """Load wine dataset and map it to binary classification for comparability.

    Class 0 is kept as 0, and classes 1/2 are merged into class 1.
    """
    ds = load_wine(as_frame=True)
    X = ds.data.copy()
    y_multiclass = ds.target.copy()
    y_binary = (y_multiclass != 0).astype(int).astype(str)
    return _process_dataframe("wine_binary", X, y_binary)



def load_adult_dataset() -> DatasetBundle:
    """Load and preprocess Adult Income dataset from OpenML."""
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    X = adult.data.copy()
    y = adult.target.copy().astype(str)
    y = y.replace({"<=50K.": "<=50K", ">50K.": ">50K"})
    return _process_dataframe("adult_income", X, y)



def load_all_datasets() -> Dict[str, DatasetBundle]:
    """Load required datasets with graceful fallback if Adult is unavailable."""
    datasets: Dict[str, DatasetBundle] = {}
    datasets["breast_cancer"] = load_breast_cancer_dataset()
    datasets["wine_binary"] = load_wine_binary_dataset()

    try:
        datasets["adult_income"] = load_adult_dataset()
    except Exception:
        # Fallback public dataset from sklearn if OpenML network is unavailable.
        fallback = load_wine(as_frame=True)
        datasets["wine_fallback_multiclass"] = _process_dataframe(
            "wine_fallback_multiclass", fallback.data.copy(), fallback.target.copy().astype(str)
        )

    return datasets



def get_dataset_names(datasets: Dict[str, DatasetBundle]) -> List[str]:
    return list(datasets.keys())
