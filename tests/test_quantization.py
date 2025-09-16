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

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exporters.coreml.quantization import (
    QuantizationMode,
    collect_coreml_calibration_samples,
    collect_torch_calibration_samples,
    quantization_metadata,
    resolve_quantization_config,
)

from .test_coreml import DummyTokenizer, build_config


def test_resolve_float_precision():
    config = resolve_quantization_config("float32")
    assert config.mode is QuantizationMode.FLOAT
    assert config.precision == "float32"
    assert config.tag == "float32"


def test_resolve_weight_quantization():
    config = resolve_quantization_config("rtn-int4")
    assert config.mode is QuantizationMode.WEIGHT_ONLY
    assert config.weight_bits == 4
    assert config.tag == "rtn-int4"


def test_resolve_activation_quantization():
    config = resolve_quantization_config("activation-int8")
    assert config.mode is QuantizationMode.ACTIVATION
    assert config.activation_bits == 8
    assert config.tag == "activation-int8"


def test_collect_coreml_calibration_samples_uses_reference_shape():
    coreml_config = build_config()
    tokenizer = DummyTokenizer()
    dummy_inputs = coreml_config.generate_dummy_inputs(tokenizer)
    samples = collect_coreml_calibration_samples(
        coreml_config,
        tokenizer,
        dummy_inputs,
        sample_count=3,
    )
    assert len(samples) == 3
    for sample in samples:
        assert "input_ids" in sample
        assert sample["input_ids"].dtype == np.int32


@pytest.mark.skipif(True, reason="Torch is not available in the test environment")
def test_collect_torch_calibration_samples_shapes():
    coreml_config = build_config()
    tokenizer = DummyTokenizer()
    tensors = collect_torch_calibration_samples(
        coreml_config,
        tokenizer,
        sample_count=2,
    )
    assert len(tensors) == 2
    for tensor in tensors:
        assert tensor.shape[0] == 1


def test_quantization_metadata_contains_mode():
    config = resolve_quantization_config("rtn-int8")
    metadata = quantization_metadata(config)
    assert metadata["co.huggingface.exporters.quantization.mode"] == QuantizationMode.WEIGHT_ONLY.value
    assert metadata["co.huggingface.exporters.quantization.weight_bits"] == "8"
