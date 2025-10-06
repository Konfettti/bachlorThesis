"""Utilities to bridge Featuretools-generated features with TabPFN models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Mapping, MutableMapping, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import featuretools as ft

DataFrameDict = Mapping[str, pd.DataFrame]
EntitySetBuilder = Callable[[DataFrameDict], ft.EntitySet]


@dataclass
class FeaturetoolsTabPFNConfig:
    """Configuration describing how to run Deep Feature Synthesis."""
    target_dataframe_name: str
    agg_primitives: Optional[List[str]] = None
    trans_primitives: Optional[List[str]] = None
    max_depth: int = 2
    verbose: bool = False
    dtype: str = "float32"


class FeaturetoolsTabPFNAdapter(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer that prepares Featuretools features for TabPFN."""

    def __init__(
        self,
        entityset_builder: EntitySetBuilder,
        config: FeaturetoolsTabPFNConfig,
    ) -> None:
        self.entityset_builder = entityset_builder
        self.config = config

    @staticmethod
    def _extract_target_ids(X: Mapping[str, object]) -> Optional[Iterable]:
        ids = X.get("target_ids") if isinstance(X, MutableMapping) else None
        if ids is None:
            return None
        return list(ids)

    @staticmethod
    def _extract_dataframes(X: Mapping[str, object]) -> DataFrameDict:
        if not isinstance(X, Mapping) or "dataframes" not in X:
            raise TypeError(
                "Expected input to be a mapping with a 'dataframes' key containing pandas DataFrames."
            )
        dataframes = X["dataframes"]
        if not isinstance(dataframes, Mapping):
            raise TypeError("'dataframes' must be a mapping from names to pandas DataFrames")
        return dataframes  # type: ignore[return-value]

    def fit(self, X: Mapping[str, object], y: Optional[np.ndarray] = None):  # type: ignore[override]
        dataframes = self._extract_dataframes(X)
        target_ids = self._extract_target_ids(X)

        entityset = self.entityset_builder(dataframes)

        feature_matrix, feature_defs = ft.dfs(
            entityset=entityset,
            target_dataframe_name=self.config.target_dataframe_name,
            agg_primitives=self.config.agg_primitives,
            trans_primitives=self.config.trans_primitives,
            max_depth=self.config.max_depth,
            verbose=self.config.verbose,
        )

        feature_matrix = feature_matrix.sort_index()
        if target_ids is not None:
            feature_matrix = feature_matrix.loc[target_ids]

        self._feature_defs = feature_defs
        self._feature_columns = feature_matrix.columns.tolist()
        self._dtype = np.dtype(self.config.dtype)
        self._training_feature_matrix = feature_matrix.astype(self._dtype)
        self._cached_target_ids = list(target_ids) if target_ids is not None else None
        return self

    def transform(self, X: Mapping[str, object]):  # type: ignore[override]
        check_is_fitted(self, "_feature_defs")

        dataframes = self._extract_dataframes(X)
        target_ids = self._extract_target_ids(X)

        if target_ids is not None and target_ids == getattr(self, "_cached_target_ids", None):
            feature_matrix = self._training_feature_matrix.copy()
        else:
            entityset = self.entityset_builder(dataframes)
            feature_matrix = ft.calculate_feature_matrix(
                features=self._feature_defs,
                entityset=entityset,
            )
            feature_matrix = feature_matrix[self._feature_columns].sort_index()
            if target_ids is not None:
                feature_matrix = feature_matrix.loc[target_ids]
            feature_matrix = feature_matrix.astype(self._dtype)

        self._cached_target_ids = list(target_ids) if target_ids is not None else None
        return feature_matrix.to_numpy()

    def get_feature_names_out(self) -> List[str]:
        check_is_fitted(self, "_feature_columns")
        return list(self._feature_columns)
