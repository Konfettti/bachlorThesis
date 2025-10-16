"""Backward compatible entry points for Featuretools-based adapters."""

from __future__ import annotations

from bachlorThesis.featuretools_adapters import (
    FeaturetoolsAdapterBase as _FeaturetoolsAdapterBase,
    FeaturetoolsAdapterConfig,
    FeaturetoolsLightGBMAdapter,
    FeaturetoolsTabPFNAdapter,
    FeaturetoolsXGBoostAdapter,
)

# Backwards compatibility alias for existing imports.
FeaturetoolsTabPFNConfig = FeaturetoolsAdapterConfig
_FeaturetoolsAdapterBase = _FeaturetoolsAdapterBase

__all__ = [
    "FeaturetoolsAdapterConfig",
    "FeaturetoolsLightGBMAdapter",
    "FeaturetoolsTabPFNAdapter",
    "FeaturetoolsTabPFNConfig",
    "FeaturetoolsXGBoostAdapter",
]
