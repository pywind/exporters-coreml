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
"""Unit tests for the simplified Core ML exporter."""

from __future__ import annotations

import numpy as np
import pytest

from exporters.coreml.config import CoreMLConfig
from exporters.coreml.features import FeaturesManager


class DummyTokenizer:
    vocab_size = 128


class DummyConfig:
    model_type = "llama"
    name_or_path = "dummy"
    vocab_size = 128
    max_position_embeddings = 64
    num_hidden_layers = 2
    num_attention_heads = 4
    num_key_value_heads = 2
    hidden_size = 16
    use_cache = True
    architectures = ["DummyForCausalLM"]
    id2label = {}


def build_config(task: str = "text-generation", use_past: bool = False) -> CoreMLConfig:
    return CoreMLConfig(DummyConfig(), task=task, use_past=use_past)


def test_invalid_task_raises():
    with pytest.raises(ValueError):
        build_config(task="sequence-classification")


def test_inputs_without_past():
    config = build_config()
    inputs = config.inputs
    assert list(inputs.keys()) == ["input_ids", "attention_mask"]
    assert all(isinstance(desc, type(next(iter(inputs.values())))) for desc in inputs.values())


def test_inputs_with_past():
    config = build_config(task="text-generation-with-past")
    inputs = config.inputs
    past_keys = [key for key in inputs.keys() if key.startswith("past_key_values")]
    assert len(past_keys) == config.num_layers * 2


def test_outputs_with_past():
    config = build_config(task="text-generation-with-past")
    outputs = config.outputs
    present_keys = [key for key in outputs.keys() if key.startswith("present_")]
    assert len(present_keys) == config.num_layers * 2


def test_generate_dummy_inputs_shapes():
    config = build_config(task="text-generation-with-past")
    dummy_inputs = config.generate_dummy_inputs(DummyTokenizer())
    assert dummy_inputs["input_ids"][0].shape == (1, min(config.max_sequence_length, 32))
    assert dummy_inputs["attention_mask"][0].dtype == np.int64
    for idx in range(config.num_layers):
        key_name = f"past_key_values_{idx}_key"
        value_name = f"past_key_values_{idx}_value"
        assert dummy_inputs[key_name][0].shape[1] == config.num_key_value_heads
        assert dummy_inputs[value_name][0].shape == dummy_inputs[key_name][0].shape


def test_flexible_outputs():
    config = build_config(task="text-generation-with-past")
    flex = config.get_flexible_outputs()
    assert "logits" in flex
    assert flex["logits"][0]["max"] == config.max_sequence_length
    present = flex[f"present_0_key"]
    assert present[0]["min"] == 0
    assert present[0]["max"] == -1


def test_features_manager_supported_models():
    supported = FeaturesManager.get_supported_features_for_model_type("llama")
    assert "text-generation" in supported
    constructor = supported["text-generation"]
    config = constructor(DummyConfig())
    assert isinstance(config, CoreMLConfig)


def test_feature_synonyms():
    assert FeaturesManager.map_from_synonym("causal-lm-with-past") == "text-generation-with-past"
    assert FeaturesManager.map_from_synonym("text-generation") == "text-generation"
