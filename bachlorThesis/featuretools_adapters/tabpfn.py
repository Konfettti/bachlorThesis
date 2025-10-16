"""Adapter that exposes Featuretools features as numpy arrays for TabPFN."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .base import FeaturetoolsAdapterBase


class FeaturetoolsTabPFNAdapter(FeaturetoolsAdapterBase):
    """Transformer that prepares Featuretools features for TabPFN."""

    def _convert_feature_matrix(
        self, feature_matrix: pd.DataFrame, X: Mapping[str, object]
    ) -> np.ndarray:
        return feature_matrix.to_numpy()
