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
"""Minimal Core ML configuration objects for large language models."""

from __future__ import annotations

import dataclasses
from collections import OrderedDict
from enum import Enum
from typing import Any, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np

try:  # pragma: no cover - optional dependency
    from transformers.utils import TensorType, is_torch_available
except ImportError:  # pragma: no cover
    class TensorType(Enum):
        PYTORCH = "pt"
        TENSORFLOW = "tf"

    def is_torch_available() -> bool:
        return False

from ..utils import logging


logger = logging.get_logger(__name__)


@dataclasses.dataclass
class InputDescription:
    """Description of a Core ML model input."""

    name: str
    description: str = ""
    is_optional: bool = False
    sequence_length: Optional[Union[int, Tuple[int, int]]] = None


@dataclasses.dataclass
class OutputDescription:
    """Description of a Core ML model output."""

    name: str
    description: str = ""
    do_softmax: Optional[bool] = None


class CoreMLConfig:
    """Configuration describing how to export a causal language model to Core ML."""

    modality = "text"
    _SUPPORTED_TASKS = {"text-generation", "text-generation-with-past"}

    def __init__(
        self,
        config: Any,
        task: str = "text-generation",
        use_past: bool = False,
    ) -> None:
        if task not in self._SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task '{task}'. Supported tasks: {sorted(self._SUPPORTED_TASKS)}")

        if not hasattr(self, "modality"):
            raise ValueError("CoreMLConfig subclasses must define the 'modality' attribute")

        if self.modality != "text":
            raise ValueError("This simplified exporter only supports text (causal language) models")

        self._config = config
        self.task = task
        self.use_past = use_past or task.endswith("-with-past")
        self.seq2seq: Optional[str] = None

    # ---------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_model_config(
        cls,
        config: Any,
        task: str = "text-generation",
        use_past: bool = False,
        seq2seq: Optional[str] = None,
    ) -> "CoreMLConfig":
        if seq2seq is not None:
            raise ValueError("Encoder/decoder exports are not supported for causal language models")
        return cls(config, task=task, use_past=use_past)

    @classmethod
    def with_past(
        cls,
        config: Any,
        task: str = "text-generation",
        seq2seq: Optional[str] = None,
    ) -> "CoreMLConfig":
        if seq2seq is not None:
            raise ValueError("Encoder/decoder exports are not supported for causal language models")
        return cls(config, task=task, use_past=True)

    # ------------------------------------------------------------------
    # Properties describing the exported interface
    # ------------------------------------------------------------------
    @property
    def inputs(self) -> "OrderedDict[str, InputDescription]":
        sequence_length = self.max_sequence_length
        inputs: "OrderedDict[str, InputDescription]" = OrderedDict()
        inputs["input_ids"] = InputDescription(
            name="input_ids",
            description="Token ids",
            sequence_length=sequence_length,
        )

        if self.uses_attention_mask:
            inputs["attention_mask"] = InputDescription(
                name="attention_mask",
                description="Attention mask",
                sequence_length=sequence_length,
            )

        if self.use_past:
            for layer in range(self.num_layers):
                inputs[f"past_key_values_{layer}_key"] = InputDescription(
                    name=f"past_key_values_{layer}_key",
                    description="Cached key tensor",
                    is_optional=True,
                )
                inputs[f"past_key_values_{layer}_value"] = InputDescription(
                    name=f"past_key_values_{layer}_value",
                    description="Cached value tensor",
                    is_optional=True,
                )
        return inputs

    @property
    def outputs(self) -> "OrderedDict[str, OutputDescription]":
        outputs: "OrderedDict[str, OutputDescription]" = OrderedDict()
        outputs["logits"] = OutputDescription(
            name="logits",
            description="Prediction scores for each token",
        )
        if self.use_past:
            for layer in range(self.num_layers):
                outputs[f"present_{layer}_key"] = OutputDescription(
                    name=f"present_{layer}_key",
                    description="Key cache for next iteration",
                )
                outputs[f"present_{layer}_value"] = OutputDescription(
                    name=f"present_{layer}_value",
                    description="Value cache for next iteration",
                )
        return outputs

    @property
    def uses_attention_mask(self) -> bool:
        return True

    @property
    def use_legacy_format(self) -> bool:
        return False

    @property
    def is_classifier(self) -> bool:
        return False

    @property
    def atol_for_validation(self) -> float:
        return 1e-4

    @property
    def short_description(self) -> str:
        model_name = getattr(self._config, "name_or_path", getattr(self._config, "model_type", "model"))
        return f"{model_name} ({self.task})"

    # ------------------------------------------------------------------
    # Model specific information helpers
    # ------------------------------------------------------------------
    @property
    def num_layers(self) -> int:
        if hasattr(self._config, "num_hidden_layers"):
            return int(self._config.num_hidden_layers)
        if hasattr(self._config, "n_layer"):
            return int(self._config.n_layer)
        raise AttributeError("Cannot determine the number of decoder layers from the model configuration")

    @property
    def num_attention_heads(self) -> int:
        if hasattr(self._config, "num_attention_heads"):
            return int(self._config.num_attention_heads)
        if hasattr(self._config, "n_head"):
            return int(self._config.n_head)
        raise AttributeError("Cannot determine number of attention heads from the model configuration")

    @property
    def num_key_value_heads(self) -> int:
        return int(getattr(self._config, "num_key_value_heads", self.num_attention_heads))

    @property
    def hidden_size(self) -> int:
        if hasattr(self._config, "hidden_size"):
            return int(self._config.hidden_size)
        if hasattr(self._config, "n_embd"):
            return int(self._config.n_embd)
        raise AttributeError("Cannot determine the hidden size from the model configuration")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def max_sequence_length(self) -> int:
        if hasattr(self._config, "max_position_embeddings"):
            return int(self._config.max_position_embeddings)
        if hasattr(self._config, "n_positions"):
            return int(self._config.n_positions)
        return 2048

    @property
    def past_sequence_length(self) -> int:
        return 1

    def get_flexible_outputs(self) -> Mapping[str, List[Mapping[str, int]]]:
        flex: Mapping[str, List[Mapping[str, int]]] = {
            "logits": [
                {
                    "axis": 1,
                    "min": 1,
                    "max": self.max_sequence_length,
                }
            ]
        }
        if self.use_past:
            for layer in range(self.num_layers):
                flex[f"present_{layer}_key"] = [
                    {
                        "axis": 2,
                        "min": 0,
                        "max": -1,
                    }
                ]
                flex[f"present_{layer}_value"] = [
                    {
                        "axis": 2,
                        "min": 0,
                        "max": -1,
                    }
                ]
        return flex

    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        if hasattr(self._config, "use_cache"):
            return {"use_cache": self.use_past}
        return None

    def get_class_labels(self) -> Iterable[str]:
        return []

    # ------------------------------------------------------------------
    # Dummy inputs
    # ------------------------------------------------------------------
    def generate_dummy_inputs(
        self,
        preprocessor: Any,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Tuple[Any, Any]]:
        batch_size = 1
        vocab_size = getattr(preprocessor, "vocab_size", getattr(self._config, "vocab_size", 32000))
        sequence_length = min(self.max_sequence_length, 32)

        input_ids = np.random.randint(0, vocab_size, (batch_size, sequence_length), dtype=np.int64)
        attention_mask = np.ones((batch_size, sequence_length), dtype=np.int64)

        dummy_inputs: Mapping[str, Tuple[Any, Any]] = OrderedDict()
        dummy_inputs["input_ids"] = (input_ids, input_ids.astype(np.int32))
        if self.uses_attention_mask:
            dummy_inputs["attention_mask"] = (attention_mask, attention_mask.astype(np.int32))

        if self.use_past:
            for layer, (key, value) in enumerate(self._generate_dummy_past_key_values(batch_size)):
                dummy_inputs[f"past_key_values_{layer}_key"] = (key, key.astype(np.float32))
                dummy_inputs[f"past_key_values_{layer}_value"] = (value, value.astype(np.float32))

        return self._convert_dummy_inputs_to_framework(dummy_inputs, framework)

    def _generate_dummy_past_key_values(self, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        seq_len = self.past_sequence_length
        key_heads = self.num_key_value_heads
        head_dim = self.hidden_size // key_heads

        if seq_len == 0:
            seq_len = 1

        shape = (batch_size, key_heads, seq_len, head_dim)
        caches: List[Tuple[np.ndarray, np.ndarray]] = []
        for _ in range(self.num_layers):
            key = np.zeros(shape, dtype=np.float32)
            value = np.zeros(shape, dtype=np.float32)
            caches.append((key, value))
        return caches

    def _convert_dummy_inputs_to_framework(self, dummy_inputs, framework):
        if framework == TensorType.PYTORCH and is_torch_available():  # pragma: no cover
            import torch

            for key, (ref_value, coreml_value) in dummy_inputs.items():
                if isinstance(ref_value, np.ndarray):
                    dummy_inputs[key] = (torch.tensor(ref_value), coreml_value)
        return dummy_inputs

    # ------------------------------------------------------------------
    # Pytorch conversion hooks
    # ------------------------------------------------------------------
    def patch_pytorch_ops(self):  # pragma: no cover - overridden by subclasses when required
        from . import models

        return models.patch_common_pytorch_ops()


__all__ = [
    "CoreMLConfig",
    "InputDescription",
    "OutputDescription",
]
