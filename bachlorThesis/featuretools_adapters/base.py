"""Core utilities for converting Featuretools outputs into model-ready data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import featuretools as ft
from pandas.api.types import is_categorical_dtype, is_object_dtype, is_string_dtype

DataFrameDict = Mapping[str, pd.DataFrame]
EntitySetBuilder = Callable[[DataFrameDict], ft.EntitySet]


@dataclass
class FeaturetoolsAdapterConfig:
    """Configuration describing how to run Deep Feature Synthesis."""

    target_dataframe_name: str
    agg_primitives: Optional[List[str]] = None
    trans_primitives: Optional[List[str]] = None
    where_primitives: Optional[List[str]] = None
    max_depth: int = 2
    training_window: Optional[str] = None
    primitive_options: Optional[Mapping[str, Any]] = None
    interesting_values: Optional[Mapping[str, Mapping[str, Sequence[Any]]]] = None
    verbose: bool = False
    dtype: str = "float32"


class FeaturetoolsAdapterBase(BaseEstimator, TransformerMixin):
    """Shared implementation for Featuretools-based transformers."""

    def __init__(
        self,
        entityset_builder: EntitySetBuilder,
        config: FeaturetoolsAdapterConfig,
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

    @staticmethod
    def _extract_labels(X: Mapping[str, object]) -> Optional[np.ndarray]:
        if not isinstance(X, MutableMapping) or "labels" not in X:
            return None
        labels = X["labels"]
        if labels is None:
            return None
        if isinstance(labels, pd.Series):
            labels = labels.to_numpy()
        labels_array = np.asarray(labels)
        if labels_array.ndim != 1:
            raise ValueError("Provided labels must be one-dimensional.")
        return labels_array

    def fit(self, X: Mapping[str, object], y: Optional[np.ndarray] = None):  # type: ignore[override]
        dataframes = self._extract_dataframes(X)
        target_ids = self._extract_target_ids(X)

        entityset = self.entityset_builder(dataframes)
        if self.config.interesting_values:
            self._apply_interesting_values(entityset, self.config.interesting_values)

        if target_ids is not None and len(target_ids) == 0:
            raise ValueError("No target_ids provided for Featuretools DFS.")

        feature_matrix, feature_defs = ft.dfs(
            entityset=entityset,
            target_dataframe_name=self.config.target_dataframe_name,
            agg_primitives=self.config.agg_primitives,
            trans_primitives=self.config.trans_primitives,
            where_primitives=self.config.where_primitives,
            max_depth=self.config.max_depth,
            training_window=self.config.training_window,
            primitive_options=self.config.primitive_options,
            verbose=self.config.verbose,
            instance_ids=target_ids,
        )

        feature_matrix = feature_matrix.sort_index()
        if target_ids is not None:
            feature_matrix = feature_matrix.loc[target_ids]

        feature_encoders = self._fit_feature_categorical_encoders(feature_matrix)
        feature_matrix = self._encode_feature_categoricals(feature_matrix, feature_encoders)
        feature_matrix = feature_matrix.astype(self.config.dtype)

        self._feature_defs = feature_defs
        self._feature_columns = feature_matrix.columns.tolist()
        self._feature_encoders = feature_encoders
        self._dtype = np.dtype(self.config.dtype)
        self._training_feature_matrix = feature_matrix
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
            if self.config.interesting_values:
                self._apply_interesting_values(entityset, self.config.interesting_values)
            feature_matrix = ft.calculate_feature_matrix(
                features=self._feature_defs,
                entityset=entityset,
                instance_ids=target_ids,
            )
            feature_matrix = feature_matrix[self._feature_columns].sort_index()
            if target_ids is not None:
                feature_matrix = feature_matrix.loc[target_ids]
            feature_matrix = self._encode_feature_categoricals(feature_matrix, self._feature_encoders)
            feature_matrix = feature_matrix.astype(self._dtype)

        drop_cols = [c for c in ("observation_id", getattr(X, "entity_col", None)) if c in feature_matrix.columns]

        if drop_cols:
            feature_matrix = feature_matrix.drop(columns=drop_cols)

        self._cached_target_ids = list(target_ids) if target_ids is not None else None
        return self._convert_feature_matrix(feature_matrix, X)

    def get_feature_names_out(self) -> List[str]:
        check_is_fitted(self, "_feature_columns")
        return list(self._feature_columns)

    def _convert_feature_matrix(
        self, feature_matrix: pd.DataFrame, X: Mapping[str, object]
    ):
        return feature_matrix

    @staticmethod
    def _apply_interesting_values(
        entityset: ft.EntitySet, interesting_values: Mapping[str, Mapping[str, Sequence[Any]]]
    ) -> None:
        for dataframe_name, columns in interesting_values.items():
            if dataframe_name not in entityset.dataframe_dict:
                raise ValueError(f"Unknown dataframe '{dataframe_name}' in interesting_values.")
            dataframe = entityset[dataframe_name]
            for column_name, values in columns.items():
                if column_name not in dataframe.columns:
                    raise ValueError(
                        f"Column '{column_name}' not found in dataframe '{dataframe_name}' for interesting_values."
                    )
                dataframe.ww.columns[column_name].interesting_values = list(values)

    @staticmethod
    def _fit_feature_categorical_encoders(feature_matrix: pd.DataFrame) -> Dict[str, Dict[Any, int]]:
        encoders: Dict[str, Dict[Any, int]] = {}
        for col in feature_matrix.columns:
            series = feature_matrix[col]
            if is_object_dtype(series) or is_categorical_dtype(series) or is_string_dtype(series):
                categories = list(pd.Categorical(series).categories)
                encoders[col] = {val: idx for idx, val in enumerate(categories)}
        return encoders

    @staticmethod
    def _encode_feature_categoricals(
        feature_matrix: pd.DataFrame, encoders: Mapping[str, Dict[Any, int]]
    ) -> pd.DataFrame:
        if not encoders:
            return feature_matrix
        df = feature_matrix.copy()
        for col, mapping in encoders.items():
            if col not in df.columns:
                continue
            df[col] = df[col].astype(object).map(mapping).astype("Int64")
        return df
