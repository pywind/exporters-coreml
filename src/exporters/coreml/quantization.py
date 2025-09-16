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
"""Minimal quantization helpers for the Core ML exporter."""

from __future__ import annotations

import dataclasses
import importlib
from dataclasses import dataclass
from typing import Dict, Union

from ..utils import logging


logger = logging.get_logger(__name__)

_ct_spec = importlib.util.find_spec("coremltools")
ct = importlib.import_module("coremltools") if _ct_spec is not None else None  # type: ignore


@dataclass
class QuantizationConfig:
    """Configuration describing the desired Core ML precision."""

    precision: str = "float16"

    @property
    def coreml_compute_precision(self):
        """Return the Core ML precision enum associated with the configuration."""

        if ct is None:  # pragma: no cover - optional dependency
            raise RuntimeError("coremltools must be installed to compute Core ML precision")
        if self.precision == "float32":
            return ct.precision.FLOAT32
        return ct.precision.FLOAT16

    def metadata_entries(self) -> Dict[str, str]:
        return {
            "co.huggingface.exporters.precision": self.precision,
            "co.huggingface.exporters.quantization": self.precision,
        }


def resolve_quantization_config(quantize: Union[str, QuantizationConfig]) -> QuantizationConfig:
    """Resolve user input into a :class:`QuantizationConfig`."""

    if isinstance(quantize, QuantizationConfig):
        return dataclasses.replace(quantize)

    value = str(quantize).lower()
    if value in {"float32", "fp32"}:
        return QuantizationConfig(precision="float32")
    if value in {"float16", "fp16"}:
        return QuantizationConfig(precision="float16")

    raise ValueError("Unsupported quantization value '{value}'. Supported values: float32, float16.".format(value=quantize))


def apply_post_training_quantization(mlmodel: "MLModel", quant_config: QuantizationConfig):
    """Return the model unchanged for float precision exports."""

    if ct is None:  # pragma: no cover - optional dependency
        logger.info("coremltools not available, skipping additional quantization steps")
        return mlmodel

    precision = quant_config.coreml_compute_precision
    if precision in {ct.precision.FLOAT32, ct.precision.FLOAT16}:
        logger.info("Skipping Core ML post-processing for float precision export")
        return mlmodel

    return mlmodel


def quantization_metadata(quant_config: QuantizationConfig) -> Dict[str, str]:
    """Metadata describing the export precision."""

    return quant_config.metadata_entries()


__all__ = [
    "QuantizationConfig",
    "apply_post_training_quantization",
    "quantization_metadata",
    "resolve_quantization_config",
]
