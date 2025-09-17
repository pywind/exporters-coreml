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
from typing import Callable, Dict, Iterable, Optional, Union

from ..utils import logging


logger = logging.get_logger(__name__)

_ct_spec = importlib.util.find_spec("coremltools")
ct = importlib.import_module("coremltools") if _ct_spec is not None else None  # type: ignore


@dataclass
class WeightCompressionConfig:
    """Describe optional Core ML weight compression to run after conversion.

    The compression utilities operate on ML Program weights without altering the
    compute precision of the exported model. Each method corresponds to a helper
    in :mod:`coremltools.compression_utils` as described in Apple's
    "Compressing ML Program Weights" guide.
    """

    method: str
    mode: Optional[str] = None
    nbits: Optional[int] = None
    threshold: Optional[float] = None
    target_percentile: Optional[float] = None
    min_elements: Optional[int] = None

    def __post_init__(self) -> None:
        method = self.method.lower()
        if method not in {"affine", "palettize", "sparsify"}:
            raise ValueError(
                "Unsupported weight compression method '{method}'. Supported values: affine, palettize, sparsify.".format(
                    method=self.method
                )
            )
        self.method = method

        if self.min_elements is not None and self.min_elements <= 0:
            raise ValueError("min_elements must be greater than zero when provided")

        if self.method == "affine":
            self._validate_mode({None, "linear", "linear_symmetric"})
            if self.nbits is not None:
                raise ValueError("'nbits' is not supported for affine weight compression")
            if self.threshold is not None or self.target_percentile is not None:
                raise ValueError("'threshold' and 'target_percentile' are not applicable to affine compression")
        elif self.method == "palettize":
            self._validate_mode({None, "uniform", "kmeans", "unique"})
            if self.nbits is not None and self.nbits not in {1, 2, 4, 6, 8}:
                raise ValueError("'nbits' must be one of {1, 2, 4, 6, 8} when palettizing weights")
            if self.threshold is not None or self.target_percentile is not None:
                raise ValueError("'threshold' and 'target_percentile' are not applicable to palettize compression")
        elif self.method == "sparsify":
            if self.mode is None:
                self.mode = "threshold_based"
            self._validate_mode({"threshold_based", "percentile_based"})
            if self.nbits is not None:
                raise ValueError("'nbits' is not supported for sparsify weight compression")
            if self.mode == "threshold_based":
                if self.threshold is not None and self.threshold < 0:
                    raise ValueError("'threshold' must be non-negative for sparsify threshold mode")
            if self.mode == "percentile_based":
                if self.target_percentile is None:
                    raise ValueError("'target_percentile' must be provided for sparsify percentile mode")
                if not (0.0 < self.target_percentile < 1.0):
                    raise ValueError("'target_percentile' must be between 0 and 1 (exclusive)")
            elif self.target_percentile is not None:
                raise ValueError("'target_percentile' is only valid for sparsify percentile mode")

    def _validate_mode(self, accepted_modes: Iterable[Optional[str]]) -> None:
        accepted = set(accepted_modes)
        if self.mode not in accepted:
            human_readable = ", ".join(sorted(m for m in accepted if m is not None))
            raise ValueError(
                "Unsupported compression mode '{mode}' for method '{method}'. Supported values: {values}.".format(
                    mode=self.mode,
                    method=self.method,
                    values=human_readable if human_readable else "<default>",
                )
            )

    def describe(self) -> str:
        """Return a concise textual representation for metadata."""

        parts = [self.method]
        if self.mode:
            parts.append(f"mode={self.mode}")
        if self.nbits is not None:
            parts.append(f"nbits={self.nbits}")
        if self.threshold is not None:
            parts.append(f"threshold={self.threshold}")
        if self.target_percentile is not None:
            parts.append(f"target_percentile={self.target_percentile}")
        if self.min_elements is not None:
            parts.append(f"min_elements={self.min_elements}")
        return ",".join(parts)

    def _build_op_selector(self) -> Optional[Callable[[object], bool]]:
        if self.min_elements is None:
            return None

        threshold = self.min_elements

        def selector(const_op: object) -> bool:  # pragma: no cover - behaviour validated via apply()
            try:
                value = const_op.val.val
            except AttributeError:
                return False
            size = getattr(value, "size", None)
            if size is None:
                return False
            return size >= threshold

        return selector

    def _compression_function(self, compression_utils) -> Callable:
        mapping = {
            "affine": "affine_quantize_weights",
            "palettize": "palettize_weights",
            "sparsify": "sparsify_weights",
        }
        name = mapping[self.method]
        if not hasattr(compression_utils, name):
            raise RuntimeError(
                "Installed coremltools version does not provide compression utility '{name}'".format(name=name)
            )
        return getattr(compression_utils, name)

    def _compression_kwargs(self) -> Dict[str, object]:
        kwargs: Dict[str, object] = {}
        if self.mode is not None:
            kwargs["mode"] = self.mode
        if self.method == "palettize" and self.nbits is not None:
            kwargs["nbits"] = self.nbits
        if self.method == "sparsify":
            if self.mode == "threshold_based" and self.threshold is not None:
                kwargs["threshold"] = self.threshold
            if self.mode == "percentile_based" and self.target_percentile is not None:
                kwargs["target_percentile"] = self.target_percentile
        op_selector = self._build_op_selector()
        if op_selector is not None:
            kwargs["op_selector"] = op_selector
        return kwargs

    def apply(self, mlmodel: "MLModel", ct_module) -> "MLModel":
        """Apply the configured weight compression using ``coremltools``."""

        if ct_module is None:  # pragma: no cover - optional dependency
            logger.info("coremltools not available, skipping weight compression")
            return mlmodel

        compression_utils = getattr(ct_module, "compression_utils", None)
        if compression_utils is None:  # pragma: no cover - depends on installed version
            logger.info("coremltools does not expose compression utilities, skipping weight compression")
            return mlmodel

        compression_fn = self._compression_function(compression_utils)
        kwargs = self._compression_kwargs()
        display_kwargs = {k: v for k, v in kwargs.items() if k != "op_selector"}
        if display_kwargs:
            logger.info(
                "Applying Core ML weight compression method '%s' with arguments %s",
                self.method,
                display_kwargs,
            )
        else:
            logger.info("Applying Core ML weight compression method '%s'", self.method)
        return compression_fn(mlmodel, **kwargs)


@dataclass
class QuantizationConfig:
    """Configuration describing the desired Core ML precision and compression."""

    precision: str = "float16"
    weight_compression: Optional[WeightCompressionConfig] = None

    @property
    def coreml_compute_precision(self):
        """Return the Core ML precision enum associated with the configuration."""

        if ct is None:  # pragma: no cover - optional dependency
            raise RuntimeError("coremltools must be installed to compute Core ML precision")
        if self.precision == "float32":
            return ct.precision.FLOAT32
        return ct.precision.FLOAT16

    def metadata_entries(self) -> Dict[str, str]:
        entries = {
            "co.huggingface.exporters.precision": self.precision,
            "co.huggingface.exporters.quantization": self.precision,
        }
        if self.weight_compression is not None:
            entries["co.huggingface.exporters.weight_compression"] = self.weight_compression.describe()
        return entries


def resolve_weight_compression_config(
    method: Optional[str],
    *,
    mode: Optional[str] = None,
    nbits: Optional[int] = None,
    threshold: Optional[float] = None,
    target_percentile: Optional[float] = None,
    min_elements: Optional[int] = None,
) -> Optional[WeightCompressionConfig]:
    """Resolve CLI arguments into a :class:`WeightCompressionConfig`."""

    if method is None:
        return None

    normalized = method.strip()
    if not normalized or normalized.lower() == "none":
        return None

    return WeightCompressionConfig(
        method=normalized,
        mode=mode,
        nbits=nbits,
        threshold=threshold,
        target_percentile=target_percentile,
        min_elements=min_elements,
    )


def resolve_quantization_config(
    quantize: Union[str, QuantizationConfig],
    *,
    weight_compression: Optional[WeightCompressionConfig] = None,
) -> QuantizationConfig:
    """Resolve user input into a :class:`QuantizationConfig`."""

    if isinstance(quantize, QuantizationConfig):
        return dataclasses.replace(quantize, weight_compression=weight_compression)

    value = str(quantize).lower()
    if value in {"float32", "fp32"}:
        return QuantizationConfig(precision="float32", weight_compression=weight_compression)
    if value in {"float16", "fp16"}:
        return QuantizationConfig(precision="float16", weight_compression=weight_compression)

    raise ValueError("Unsupported quantization value '{value}'. Supported values: float32, float16.".format(value=quantize))


def apply_post_training_quantization(mlmodel: "MLModel", quant_config: QuantizationConfig):
    """Return the model unchanged for float precision exports."""

    if ct is None:  # pragma: no cover - optional dependency
        logger.info("coremltools not available, skipping additional quantization steps")
        return mlmodel

    if quant_config.weight_compression is not None:
        mlmodel = quant_config.weight_compression.apply(mlmodel, ct)

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
    "WeightCompressionConfig",
    "apply_post_training_quantization",
    "quantization_metadata",
    "resolve_weight_compression_config",
    "resolve_quantization_config",
]
