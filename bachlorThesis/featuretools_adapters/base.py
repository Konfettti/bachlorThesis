"""Core utilities for converting Featuretools outputs into model-ready data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import featuretools as ft

DataFrameDict = Mapping[str, pd.DataFrame]
EntitySetBuilder = Callable[[DataFrameDict], ft.EntitySet]


@dataclass
class FeaturetoolsAdapterConfig:
    """Configuration describing how to run Deep Feature Synthesis."""

    target_dataframe_name: str
    agg_primitives: Optional[List[str]] = None
    agg_primitive_dataframes: Optional[List[str]] = None
    agg_primitive_include_columns: Optional[Dict[str, List[str]]] = None
    agg_primitive_ignore_columns: Optional[Dict[str, List[str]]] = None
    agg_primitive_options: Optional[Dict[str, Dict[str, object]]] = None
    trans_primitives: Optional[List[str]] = None
    max_depth: int = 2
    training_window: Optional[str] = None
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

    @staticmethod
    def _fit_categorical_encoders(df: pd.DataFrame) -> Dict[str, CategoricalDtype]:
        encoders: Dict[str, CategoricalDtype] = {}
        categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns
        for col in categorical_cols:
            encoders[col] = CategoricalDtype(categories=pd.Categorical(df[col]).categories)
        return encoders

    @staticmethod
    def _encode_categoricals(
        df: pd.DataFrame, encoders: Optional[Dict[str, CategoricalDtype]] = None
    ) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns
        for col in categorical_cols:
            dtype = encoders.get(col) if encoders is not None else None
            if dtype is None:
                codes = pd.Categorical(df[col]).codes
            else:
                codes = pd.Categorical(df[col], categories=dtype.categories).codes
            df[col] = pd.Series(codes, index=df.index).replace(-1, pd.NA).astype("Int64")
        return df

    def fit(self, X: Mapping[str, object], y: Optional[np.ndarray] = None):  # type: ignore[override]
        dataframes = self._extract_dataframes(X)
        target_ids = self._extract_target_ids(X)

        entityset = self.entityset_builder(dataframes)

        primitive_options = None
        if self.config.agg_primitives:
            primitive_options = (
                {key: dict(value) for key, value in self.config.agg_primitive_options.items()}
                if self.config.agg_primitive_options
                else {}
            )
            has_scoping = any(
                (
                    self.config.agg_primitive_dataframes,
                    self.config.agg_primitive_include_columns,
                    self.config.agg_primitive_ignore_columns,
                )
            )
            if has_scoping:
                for primitive in self.config.agg_primitives:
                    options = primitive_options.setdefault(primitive, {})
                    if self.config.agg_primitive_dataframes and "include_dataframes" not in options:
                        options["include_dataframes"] = list(self.config.agg_primitive_dataframes)
                    if self.config.agg_primitive_include_columns and "include_columns" not in options:
                        options["include_columns"] = self.config.agg_primitive_include_columns
                    if self.config.agg_primitive_ignore_columns and "ignore_columns" not in options:
                        options["ignore_columns"] = self.config.agg_primitive_ignore_columns
            primitive_options = primitive_options or None

        cutoff_time = X.get("cutoff_time") if isinstance(X, Mapping) else None

        feature_matrix, feature_defs = ft.dfs(
            entityset=entityset,
            target_dataframe_name=self.config.target_dataframe_name,
            cutoff_time=cutoff_time,
            agg_primitives=self.config.agg_primitives,
            trans_primitives=self.config.trans_primitives,
            primitive_options=primitive_options,
            training_window=self.config.training_window,
            max_depth=self.config.max_depth,
            verbose=self.config.verbose,
        )

        feature_matrix = feature_matrix.sort_index()
        if target_ids is not None:
            feature_matrix = feature_matrix.loc[target_ids]

        self._categorical_encoders = self._fit_categorical_encoders(feature_matrix)
        feature_matrix = self._encode_categoricals(feature_matrix, self._categorical_encoders)
        feature_matrix = feature_matrix.astype(self.config.dtype)

        self._feature_defs = feature_defs
        self._feature_columns = feature_matrix.columns.tolist()
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
            feature_matrix = ft.calculate_feature_matrix(
                features=self._feature_defs,
                entityset=entityset,
            )
            feature_matrix = feature_matrix[self._feature_columns].sort_index()
            if target_ids is not None:
                feature_matrix = feature_matrix.loc[target_ids]
            feature_matrix = self._encode_categoricals(
                feature_matrix, getattr(self, "_categorical_encoders", None)
            )
            feature_matrix = feature_matrix.astype(self._dtype)

        entity_col = X.get("entity_col") if isinstance(X, Mapping) else getattr(X, "entity_col", None)
        drop_cols = [c for c in ("observation_id", entity_col) if c in feature_matrix.columns]

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
