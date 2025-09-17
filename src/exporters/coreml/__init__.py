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
"""Public API for the Core ML exporter."""

from __future__ import annotations

from .config import CoreMLConfig

try:  # pragma: no cover - optional dependency
    from .convert import export
except ImportError:  # pragma: no cover
    def export(*args, **kwargs):
        raise ImportError("The Core ML exporter requires PyTorch and coremltools to be installed")

try:  # pragma: no cover - optional dependency
    from .validate import validate_model_outputs
except ImportError:  # pragma: no cover
    def validate_model_outputs(*args, **kwargs):
        raise ImportError("The Core ML exporter requires coremltools to be installed for validation")

__all__ = ["CoreMLConfig", "export", "validate_model_outputs"]
