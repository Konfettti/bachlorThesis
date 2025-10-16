"""Adapter that returns XGBoost ``DMatrix`` objects from Featuretools features."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from .base import FeaturetoolsAdapterBase


class FeaturetoolsXGBoostAdapter(FeaturetoolsAdapterBase):
    """Transformer that prepares Featuretools features for XGBoost."""

    def _convert_feature_matrix(
        self, feature_matrix: pd.DataFrame, X: Mapping[str, object]
    ):
        try:
            import xgboost as xgb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "xgboost is required to use FeaturetoolsXGBoostAdapter."
            ) from exc

        labels = self._extract_labels(X)
        if labels is not None and len(labels) != len(feature_matrix):
            raise ValueError("Number of labels must match the number of rows in the feature matrix.")

        return xgb.DMatrix(
            data=feature_matrix.to_numpy(),
            label=labels,
            feature_names=self.get_feature_names_out(),
        )
