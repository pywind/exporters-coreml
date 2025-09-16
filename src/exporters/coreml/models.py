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
"""Core ML configuration specialisations for popular large language models."""

from __future__ import annotations

from typing import Dict

from .config import CoreMLConfig


def patch_common_pytorch_ops() -> Dict[str, object]:  # pragma: no cover - placeholder for future patches
    """Return a dictionary of additional PyTorch operator converters.

    The latest versions of `coremltools` already support most of the operations
    used by modern decoder-only architectures. This helper acts as an extension
    point for model-specific workarounds whenever coremltools lacks a converter
    for a particular operator.
    """

    return {}


class LlamaCoreMLConfig(CoreMLConfig):
    modality = "text"


class MistralCoreMLConfig(CoreMLConfig):
    modality = "text"


class QwenCoreMLConfig(CoreMLConfig):
    modality = "text"


class Qwen2CoreMLConfig(CoreMLConfig):
    modality = "text"


class Phi3CoreMLConfig(CoreMLConfig):
    modality = "text"


class GPT2CoreMLConfig(CoreMLConfig):
    modality = "text"

    @property
    def max_sequence_length(self) -> int:
        if hasattr(self._config, "n_positions"):
            return int(self._config.n_positions)
        return super().max_sequence_length


__all__ = [
    "GPT2CoreMLConfig",
    "LlamaCoreMLConfig",
    "MistralCoreMLConfig",
    "Phi3CoreMLConfig",
    "QwenCoreMLConfig",
    "Qwen2CoreMLConfig",
    "patch_common_pytorch_ops",
]
