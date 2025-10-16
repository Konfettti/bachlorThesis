"""Adapter that returns LightGBM ``Dataset`` objects from Featuretools features."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from .base import FeaturetoolsAdapterBase


class FeaturetoolsLightGBMAdapter(FeaturetoolsAdapterBase):
    """Transformer that prepares Featuretools features for LightGBM."""

    def _convert_feature_matrix(
        self, feature_matrix: pd.DataFrame, X: Mapping[str, object]
    ):
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "lightgbm is required to use FeaturetoolsLightGBMAdapter."
            ) from exc

        labels = self._extract_labels(X)
        if labels is not None and len(labels) != len(feature_matrix):
            raise ValueError("Number of labels must match the number of rows in the feature matrix.")

        dataset = lgb.Dataset(
            data=feature_matrix.to_numpy(),
            label=labels,
            feature_name=self.get_feature_names_out(),
            free_raw_data=False,
        )
        return dataset
