# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the quantization helpers."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exporters.coreml.quantization import (  # noqa: E402
    QuantizationConfig,
    WeightCompressionConfig,
    apply_post_training_quantization,
    quantization_metadata,
    resolve_quantization_config,
    resolve_weight_compression_config,
)


class DummyModel:
    pass


def test_resolve_float_precision():
    config = resolve_quantization_config("float32")
    assert isinstance(config, QuantizationConfig)
    assert config.precision == "float32"


def test_default_precision_is_float16():
    config = resolve_quantization_config("float16")
    assert config.precision == "float16"


@pytest.mark.parametrize("value", ["float32", "float16", "fp32", "fp16"])
def test_metadata_contains_precision(value):
    config = resolve_quantization_config(value)
    metadata = quantization_metadata(config)
    assert metadata["co.huggingface.exporters.precision"] == config.precision
    assert metadata["co.huggingface.exporters.quantization"] == config.precision


def test_apply_post_training_quantization_is_noop_without_coremltools(monkeypatch):
    config = resolve_quantization_config("float16")

    # Simulate absence of coremltools by forcing the module attribute to None.
    monkeypatch.setattr("exporters.coreml.quantization.ct", None, raising=False)

    model = DummyModel()
    assert apply_post_training_quantization(model, config) is model


def test_invalid_quantization_raises():
    with pytest.raises(ValueError):
        resolve_quantization_config("int8")


def test_resolve_weight_compression_none():
    assert resolve_weight_compression_config(None) is None
    assert resolve_weight_compression_config("none") is None


def test_invalid_weight_compression_method():
    with pytest.raises(ValueError):
        WeightCompressionConfig(method="unknown")


def test_palettize_requires_valid_nbits():
    with pytest.raises(ValueError):
        WeightCompressionConfig(method="palettize", nbits=3)


def test_sparsify_percentile_validation():
    with pytest.raises(ValueError):
        WeightCompressionConfig(method="sparsify", mode="percentile_based", target_percentile=1.5)


def test_weight_compression_metadata_entry():
    compression = WeightCompressionConfig(method="palettize", nbits=4, mode="kmeans")
    config = resolve_quantization_config("float16", weight_compression=compression)
    metadata = quantization_metadata(config)
    assert metadata["co.huggingface.exporters.weight_compression"].startswith("palettize")


def test_apply_weight_compression(monkeypatch):
    class DummyCompressionUtils:
        def __init__(self):
            self.calls = []

        def affine_quantize_weights(self, model, **kwargs):
            self.calls.append((model, kwargs))
            return ("compressed", kwargs)

    utils = DummyCompressionUtils()
    dummy_ct = type(
        "DummyCT",
        (),
        {
            "precision": type("DummyPrecision", (), {"FLOAT16": "float16", "FLOAT32": "float32"}),
            "compression_utils": utils,
        },
    )()
    monkeypatch.setattr("exporters.coreml.quantization.ct", dummy_ct, raising=False)

    compression = WeightCompressionConfig(method="affine", mode="linear")
    config = resolve_quantization_config("float16", weight_compression=compression)
    model = DummyModel()
    result = apply_post_training_quantization(model, config)

    assert result[0] == "compressed"
    assert utils.calls == [(model, {"mode": "linear"})]
