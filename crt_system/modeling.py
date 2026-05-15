from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline


@dataclass
class CRTModelBundle:
    model: Pipeline
    feature_columns: List[str]
    target_column: str
    metrics: Dict[str, float]

    def predict_proba(self, dataframe: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(dataframe[self.feature_columns])[:, 1]

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "CRTModelBundle":
        return joblib.load(path)


def _build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.03,
                    max_depth=4,
                    max_iter=400,
                    min_samples_leaf=25,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42,
                ),
            ),
        ]
    )


def _score_fold(model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series) -> Dict[str, float]:
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_valid)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_valid, prediction),
        "f1": f1_score(y_valid, prediction, zero_division=0),
    }
    if len(np.unique(y_valid)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_valid, probability)
        metrics["average_precision"] = average_precision_score(y_valid, probability)
    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan
    return metrics


def train_crt_model(dataframe: pd.DataFrame, target_column: str = "label", feature_columns: Optional[Sequence[str]] = None, n_splits: int = 5) -> CRTModelBundle:
    if feature_columns is None:
        excluded = {target_column, "setup_id", "exit_reason", "target_mode"}
        feature_columns = [column for column in dataframe.columns if column not in excluded and pd.api.types.is_numeric_dtype(dataframe[column])]
    feature_columns = list(feature_columns)
    dataset = dataframe.dropna(subset=[target_column]).copy()
    x = dataset[feature_columns]
    y = dataset[target_column].astype(int)

    fold_metrics: List[Dict[str, float]] = []
    if len(dataset) >= 4:
        split_count = min(n_splits, len(dataset) - 1)
        if split_count >= 2:
            splitter = TimeSeriesSplit(n_splits=split_count)
            for train_index, valid_index in splitter.split(x):
                model = _build_model()
                fold_metrics.append(_score_fold(model, x.iloc[train_index], y.iloc[train_index], x.iloc[valid_index], y.iloc[valid_index]))

    final_model = _build_model()
    final_model.fit(x, y)
    probability = final_model.predict_proba(x)[:, 1]
    final_metrics = {
        "train_accuracy": accuracy_score(y, (probability >= 0.5).astype(int)),
        "train_f1": f1_score(y, (probability >= 0.5).astype(int), zero_division=0),
    }
    if len(np.unique(y)) > 1:
        final_metrics["train_roc_auc"] = roc_auc_score(y, probability)
        final_metrics["train_average_precision"] = average_precision_score(y, probability)
    else:
        final_metrics["train_roc_auc"] = np.nan
        final_metrics["train_average_precision"] = np.nan

    if fold_metrics:
        final_metrics["cv_accuracy"] = float(np.nanmean([fold["accuracy"] for fold in fold_metrics]))
        final_metrics["cv_f1"] = float(np.nanmean([fold["f1"] for fold in fold_metrics]))
        final_metrics["cv_roc_auc"] = float(np.nanmean([fold["roc_auc"] for fold in fold_metrics]))
        final_metrics["cv_average_precision"] = float(np.nanmean([fold["average_precision"] for fold in fold_metrics]))

    return CRTModelBundle(model=final_model, feature_columns=feature_columns, target_column=target_column, metrics=final_metrics)
