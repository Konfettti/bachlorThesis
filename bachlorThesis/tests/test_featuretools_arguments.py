from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock

import pandas as pd

import featuretools as ft

from bachlorThesis.examples.featuretools_model_relbench_pipeline import (
    _load_json_argument,
    parse_args,
)
from bachlorThesis.featuretools_adapters import FeaturetoolsAdapterConfig, FeaturetoolsTabPFNAdapter


class FeaturetoolsArgumentTests(unittest.TestCase):
    def test_parse_args_accepts_new_flags(self) -> None:
        args = parse_args(
            [
                "--dataset",
                "rel-event",
                "--agg-primitives",
                "sum",
                "--trans-primitives",
                "month",
                "--where-primitives",
                "gt_mean",
                "--training-window",
                "30d",
                "--primitive-options-json",
                '{"sum": {"include_dataframes": ["transactions"]}}',
                "--interesting-values-json",
                '{"customers": {"country": ["DE"]}}',
            ]
        )

        self.assertEqual(args.where_primitives, ["gt_mean"])
        self.assertEqual(args.training_window, "30d")
        self.assertEqual(
            args.primitive_options_json, '{"sum": {"include_dataframes": ["transactions"]}}'
        )
        self.assertEqual(args.interesting_values_json, '{"customers": {"country": ["DE"]}}')

    def test_load_json_argument_accepts_file_and_inline(self) -> None:
        inline = _load_json_argument('{"a": 1}', "--primitive-options-json")
        self.assertEqual(inline, {"a": 1})

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp:
            json.dump({"b": {"c": 2}}, tmp)
            tmp.flush()
            file_loaded = _load_json_argument(tmp.name, "--interesting-values-json")
        self.assertEqual(file_loaded, {"b": {"c": 2}})

        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            _load_json_argument("{not-json", "--primitive-options-json")
        with self.assertRaisesRegex(ValueError, "must decode to a JSON object"):
            _load_json_argument("[]", "--interesting-values-json")

    def test_adapter_forwards_new_dfs_options(self) -> None:
        dataframes = {
            "customers": pd.DataFrame({"customer_id": [1, 2], "country": ["DE", "US"]}),
        }

        def builder(dfs):
            es = ft.EntitySet(id="customer_data")
            es = es.add_dataframe(dataframe_name="customers", dataframe=dfs["customers"], index="customer_id")
            return es

        cfg = FeaturetoolsAdapterConfig(
            target_dataframe_name="customers",
            agg_primitives=["count"],
            trans_primitives=["month"],
            where_primitives=["gt_mean"],
            training_window="30d",
            primitive_options={"count": {"include_dataframes": ["customers"]}},
            interesting_values={"customers": {"country": ["DE"]}},
            verbose=False,
        )
        adapter = FeaturetoolsTabPFNAdapter(entityset_builder=builder, config=cfg)

        def fake_dfs(**kwargs):
            entityset = kwargs["entityset"]
            column = entityset["customers"].ww.columns["country"]
            self.assertEqual(column.interesting_values, ["DE"])
            self.assertEqual(kwargs["primitive_options"], cfg.primitive_options)
            self.assertEqual(kwargs["where_primitives"], cfg.where_primitives)
            self.assertEqual(kwargs["training_window"], cfg.training_window)
            self.assertEqual(kwargs["agg_primitives"], cfg.agg_primitives)
            return (
                pd.DataFrame({"feature": [1.0, 2.0]}, index=pd.Index([1, 2], name="customer_id")),
                [],
            )

        with mock.patch("bachlorThesis.featuretools_adapters.base.ft.dfs", side_effect=fake_dfs) as mock_dfs:
            adapter.fit({"dataframes": dataframes})
            mock_dfs.assert_called_once()

    def test_adapter_encodes_categoricals_after_dfs(self) -> None:
        dataframes = {
            "customers": pd.DataFrame({"customer_id": [1, 2], "country": ["DE", "US"]}),
        }

        def builder(dfs):
            es = ft.EntitySet(id="customer_data")
            es = es.add_dataframe(dataframe_name="customers", dataframe=dfs["customers"], index="customer_id")
            return es

        cfg = FeaturetoolsAdapterConfig(
            target_dataframe_name="customers",
            agg_primitives=["count"],
            trans_primitives=None,
            verbose=False,
        )
        adapter = FeaturetoolsTabPFNAdapter(entityset_builder=builder, config=cfg)

        def fake_dfs(**kwargs):
            entityset = kwargs["entityset"]
            expected_country = pd.Series(["DE", "US"], dtype="string", index=pd.Index([1, 2]), name="country")
            pd.testing.assert_series_equal(entityset["customers"]["country"], expected_country)
            fm = pd.DataFrame(
                {
                    "COUNTRY": pd.Series(
                        ["DE", "US"], index=pd.Index([1, 2], name="customer_id"), dtype="category"
                    ),
                    "COUNT(transactions)": pd.Series([1, 2], index=pd.Index([1, 2], name="customer_id")),
                },
                index=pd.Index([1, 2], name="customer_id"),
            )
            return fm, []

        with mock.patch("bachlorThesis.featuretools_adapters.base.ft.dfs", side_effect=fake_dfs):
            adapter.fit({"dataframes": dataframes})
            feature_matrix = adapter._training_feature_matrix  # type: ignore[attr-defined]
            self.assertTrue(pd.api.types.is_float_dtype(feature_matrix["COUNTRY"]))
            self.assertListEqual(feature_matrix["COUNTRY"].tolist(), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
