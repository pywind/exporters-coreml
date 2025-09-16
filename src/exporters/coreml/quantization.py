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
"""Quantization helpers for the Core ML exporter."""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - optional dependency
    import coremltools as ct
except ImportError:  # pragma: no cover
    ct = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from transformers.utils import TensorType, is_torch_available
except ImportError:  # pragma: no cover

    class TensorType(Enum):
        PYTORCH = "pt"

    def is_torch_available() -> bool:
        return False

from .config import CoreMLConfig
from ..utils import logging


if TYPE_CHECKING:  # pragma: no cover - type checking only
    import torch
    from coremltools.models import MLModel


logger = logging.get_logger(__name__)


class QuantizationMode(str, Enum):
    """Enumeration of supported quantization strategies."""

    FLOAT = "float"
    WEIGHT_ONLY = "weight-only"
    ACTIVATION = "activation"
    GPTQ = "gptq"
    QAT = "qat"


@dataclass
class QuantizationConfig:
    """Quantization parameters resolved from user inputs."""

    mode: QuantizationMode = QuantizationMode.FLOAT
    precision: str = "float16"
    weight_bits: Optional[int] = None
    activation_bits: Optional[int] = None
    calibration_samples: int = 128
    calibration_prompts: Optional[Path] = None
    gptq_block_size: int = 128
    qat_checkpoint: Optional[Path] = None
    tag: str = field(default="float16")

    @property
    def needs_coreml_calibration(self) -> bool:
        return self.mode == QuantizationMode.ACTIVATION

    @property
    def needs_torch_calibration(self) -> bool:
        return self.mode == QuantizationMode.GPTQ

    @property
    def requires_weight_postprocess(self) -> bool:
        return self.mode in {
            QuantizationMode.WEIGHT_ONLY,
            QuantizationMode.GPTQ,
            QuantizationMode.QAT,
        }

    @property
    def coreml_compute_precision(self):
        if ct is None:  # pragma: no cover - import guard
            raise RuntimeError("coremltools must be installed to compute Core ML precision")
        if self.precision == "float32":
            return ct.precision.FLOAT32
        return ct.precision.FLOAT16

    def metadata_entries(self) -> Dict[str, str]:
        entries = {
            "co.huggingface.exporters.precision": self.precision,
            "co.huggingface.exporters.quantization": self.tag,
            "co.huggingface.exporters.quantization.mode": self.mode.value,
        }
        if self.weight_bits is not None:
            entries["co.huggingface.exporters.quantization.weight_bits"] = str(self.weight_bits)
        if self.activation_bits is not None:
            entries["co.huggingface.exporters.quantization.activation_bits"] = str(
                self.activation_bits
            )
        return entries


def resolve_quantization_config(quantize: Union[str, QuantizationConfig]) -> QuantizationConfig:
    """Resolve a user supplied quantization description into a :class:`QuantizationConfig`."""

    if isinstance(quantize, QuantizationConfig):
        return dataclasses.replace(quantize)
    return _parse_quantize_argument(str(quantize))


def _parse_quantize_argument(raw_value: str) -> QuantizationConfig:
    value = raw_value.lower()

    if value in {"float32", "fp32"}:
        return QuantizationConfig(mode=QuantizationMode.FLOAT, precision="float32", tag="float32")
    if value in {"float16", "fp16"}:
        return QuantizationConfig(mode=QuantizationMode.FLOAT, precision="float16", tag="float16")

    weight_match = re.fullmatch(r"(rtn|weight|w)(?:[-_]?int)?(4|8)", value)
    if weight_match:
        bits = int(weight_match.group(2))
        return QuantizationConfig(
            mode=QuantizationMode.WEIGHT_ONLY,
            precision="float16",
            weight_bits=bits,
            tag=f"rtn-int{bits}",
        )

    activation_match = re.fullmatch(r"(activation|act)(?:[-_]?int)?(8)", value)
    if activation_match:
        bits = int(activation_match.group(2))
        return QuantizationConfig(
            mode=QuantizationMode.ACTIVATION,
            precision="float16",
            activation_bits=bits,
            tag=f"activation-int{bits}",
        )

    gptq_match = re.fullmatch(r"gptq(?:[-_]?int)?(4|8)", value)
    if gptq_match:
        bits = int(gptq_match.group(1))
        return QuantizationConfig(
            mode=QuantizationMode.GPTQ,
            precision="float16",
            weight_bits=bits,
            tag=f"gptq-int{bits}",
        )

    qat_match = re.fullmatch(r"qat(?:[-_]?int)?(4|8)", value)
    if qat_match:
        bits = int(qat_match.group(1))
        return QuantizationConfig(
            mode=QuantizationMode.QAT,
            precision="float16",
            weight_bits=bits,
            tag=f"qat-int{bits}",
        )

    raise ValueError(
        "Unsupported quantization mode '{value}'. Supported values include "
        "float16, float32, rtn-int4, rtn-int8, activation-int8, gptq-int4 and qat-int8.".format(value=raw_value)
    )


def _load_prompts(prompts_path: Optional[Path], limit: int) -> List[str]:
    if prompts_path is None:
        return []
    prompts: List[str] = []
    try:
        with prompts_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(prompts) >= limit:
                    break
                text = line.strip()
                if text:
                    prompts.append(text)
    except OSError as error:  # pragma: no cover - filesystem dependent
        logger.warning("Unable to read calibration prompts from %s: %s", prompts_path, error)
    return prompts


def _reshape_like(sample: np.ndarray, reference: np.ndarray) -> Optional[np.ndarray]:
    if sample.shape == reference.shape:
        return sample.astype(reference.dtype, copy=False)
    if sample.ndim != reference.ndim:
        return None
    slices = tuple(slice(0, min(dim_sample, dim_ref)) for dim_sample, dim_ref in zip(sample.shape, reference.shape))
    trimmed = sample[slices]
    if trimmed.shape != reference.shape:
        padded = np.zeros(reference.shape, dtype=reference.dtype)
        padded[slices] = trimmed.astype(reference.dtype, copy=False)
        return padded
    return trimmed.astype(reference.dtype, copy=False)


def _encode_prompt_numpy(preprocessor, prompt: str, max_length: int) -> Optional[Dict[str, np.ndarray]]:
    if preprocessor is None or not hasattr(preprocessor, "__call__"):
        return None
    kwargs = {"truncation": True, "max_length": max_length, "padding": "max_length"}
    for return_tensors in ("np", "pt"):
        try:
            encoded = preprocessor(prompt, return_tensors=return_tensors, **kwargs)
        except Exception:  # pragma: no cover - depends on preprocessor type
            continue
        if return_tensors == "np":
            return {key: np.asarray(value) for key, value in encoded.items()}
        if return_tensors == "pt" and is_torch_available():
            import torch

            result = {}
            for key, value in encoded.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.detach().cpu().numpy()
            return result
    return None


def collect_coreml_calibration_samples(
    config: CoreMLConfig,
    preprocessor,
    reference_inputs: Mapping[str, Tuple[object, np.ndarray]],
    sample_count: int,
    *,
    prompts_path: Optional[Path] = None,
) -> List[Dict[str, np.ndarray]]:
    """Generate calibration samples for activation quantization."""

    if sample_count <= 0:
        return []

    prompts = _load_prompts(prompts_path, sample_count)
    reference_numpy = {
        name: value
        for name, (_, value) in reference_inputs.items()
        if not name.startswith("past_key_values")
    }

    samples: List[Dict[str, np.ndarray]] = []
    for index in range(sample_count):
        sample: Dict[str, np.ndarray] = {}
        if index < len(prompts):
            encoded = _encode_prompt_numpy(preprocessor, prompts[index], config.max_sequence_length)
            if encoded:
                for name, reference_value in reference_numpy.items():
                    if name in encoded:
                        reshaped = _reshape_like(np.asarray(encoded[name]), reference_value)
                        if reshaped is not None:
                            sample[name] = reshaped
                if sample:
                    samples.append(sample)
                    continue

        dummy = config.generate_dummy_inputs(preprocessor)
        for name, (_, value) in dummy.items():
            if name.startswith("past_key_values"):
                continue
            sample[name] = np.asarray(value)
        samples.append(sample)

    return samples


def _encode_prompt_torch(preprocessor, prompt: str, max_length: int):
    if not is_torch_available():  # pragma: no cover - dependent on torch installation
        return None
    if preprocessor is None or not hasattr(preprocessor, "__call__"):
        return None
    kwargs = {"truncation": True, "max_length": max_length, "padding": "max_length", "return_tensors": "pt"}
    try:
        encoded = preprocessor(prompt, **kwargs)
    except Exception:  # pragma: no cover
        return None
    input_ids = encoded.get("input_ids")
    return input_ids


def collect_torch_calibration_samples(
    config: CoreMLConfig,
    preprocessor,
    sample_count: int,
    *,
    prompts_path: Optional[Path] = None,
) -> List["torch.Tensor"]:
    """Collect torch tensors for GPTQ calibration."""

    if not is_torch_available() or sample_count <= 0:
        return []

    import torch

    prompts = _load_prompts(prompts_path, sample_count)
    samples: List[torch.Tensor] = []
    for index in range(sample_count):
        tensor = None
        if index < len(prompts):
            tensor = _encode_prompt_torch(preprocessor, prompts[index], config.max_sequence_length)
        if tensor is None:
            dummy = config.generate_dummy_inputs(preprocessor, framework=TensorType.PYTORCH)
            tensor = dummy["input_ids"][0]
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        samples.append(tensor.detach().clone())
    return samples


def prepare_torch_model_for_quantization(
    model,
    quant_config: QuantizationConfig,
    calibration_dataset: Sequence,
) -> None:
    """Apply algorithm-specific preprocessing on the PyTorch model prior to conversion."""

    if quant_config.mode == QuantizationMode.GPTQ:
        if not calibration_dataset:
            raise ValueError("GPTQ quantization requires calibration samples")
        _run_gptq(model, quant_config, calibration_dataset)
    elif quant_config.mode == QuantizationMode.QAT:
        _load_qat_checkpoint(model, quant_config)


def _load_qat_checkpoint(model, quant_config: QuantizationConfig) -> None:
    if quant_config.qat_checkpoint is None:
        raise ValueError("A quantization-aware training checkpoint must be provided via --qat-checkpoint")
    if not is_torch_available():  # pragma: no cover - dependent on torch
        raise RuntimeError("Quantization-aware training exports require PyTorch to be installed")

    import torch

    checkpoint_path = Path(quant_config.qat_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"QAT checkpoint not found: {checkpoint_path}")

    logger.info("Loading QAT checkpoint from %s", checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:  # pragma: no cover - depends on checkpoint
        logger.warning("Missing QAT parameters: %s", ", ".join(missing))
    if unexpected:  # pragma: no cover - depends on checkpoint
        logger.warning("Unexpected QAT parameters: %s", ", ".join(unexpected))
    model.eval()


def _run_gptq(model, quant_config: QuantizationConfig, calibration_dataset: Sequence) -> None:
    if not is_torch_available():  # pragma: no cover - dependent on torch
        raise RuntimeError("GPTQ quantization requires PyTorch to be installed")

    try:
        from coremltools.optimize.torch.layerwise_compression import LayerwiseCompressor, LayerwiseCompressorConfig
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("coremltools.optimize.torch.layerwise_compression is required for GPTQ") from error

    import torch

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    model.eval()

    config_dict = {
        "global_config": {
            "algorithm": "gptq",
            "weight_dtype": f"int{quant_config.weight_bits}",
            "granularity": "per_channel",
            "block_size": quant_config.gptq_block_size,
        },
        "input_cacher": "gpt",
        "calibration_nsamples": min(len(calibration_dataset), quant_config.calibration_samples),
    }
    compressor_config = LayerwiseCompressorConfig.from_dict(config_dict)
    compressor = LayerwiseCompressor(model, compressor_config)

    class _ListLoader:
        def __init__(self, tensors: Sequence[torch.Tensor]):
            self._tensors = [tensor.detach().clone() for tensor in tensors]

        def __iter__(self):
            for tensor in self._tensors:
                yield tensor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "Running GPTQ compression (%s-bit, block size %s) with %d calibration samples",
        quant_config.weight_bits,
        quant_config.gptq_block_size,
        config_dict["calibration_nsamples"],
    )
    with torch.no_grad():
        compressor.compress(_ListLoader(calibration_dataset), device=device, inplace=True)
    model.cpu()


def apply_post_training_quantization(
    mlmodel: "MLModel",
    quant_config: QuantizationConfig,
    calibration_data: Optional[Sequence[Mapping[str, np.ndarray]]] = None,
) -> "MLModel":
    """Apply Core ML side compression after conversion."""

    if quant_config.mode == QuantizationMode.FLOAT:
        return mlmodel

    if quant_config.mode == QuantizationMode.ACTIVATION:
        if not calibration_data:
            raise ValueError(
                "Activation quantization requires calibration samples. Provide --calibration-prompts or increase calibration-limit."
            )
        return _apply_activation_quantization(mlmodel, quant_config, calibration_data)

    if quant_config.requires_weight_postprocess:
        return _apply_weight_quantization(mlmodel, quant_config)

    return mlmodel


def _apply_weight_quantization(mlmodel: "MLModel", quant_config: QuantizationConfig) -> "MLModel":
    try:
        from coremltools.optimize import coreml as cmt_coreml
        from coremltools.optimize.coreml import _config as cmt_config
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("coremltools.optimize.coreml is required for weight quantization") from error

    dtype = f"int{quant_config.weight_bits}"
    block_size = 32 if quant_config.weight_bits == 4 else None
    global_config = cmt_coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype=dtype,
        granularity=cmt_config.CompressionGranularity.PER_CHANNEL,
        block_size=block_size,
    )
    optimization_config = cmt_coreml.OptimizationConfig(global_config=global_config)
    logger.info("Applying linear weight quantization (%s)", dtype)
    return cmt_coreml.linear_quantize_weights(mlmodel, optimization_config)


def _apply_activation_quantization(
    mlmodel: "MLModel",
    quant_config: QuantizationConfig,
    calibration_data: Sequence[Mapping[str, np.ndarray]],
) -> "MLModel":
    try:
        from coremltools.optimize import coreml as cmt_coreml
        from coremltools.optimize.coreml import _config as cmt_config
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("coremltools.optimize.coreml is required for activation quantization") from error

    dtype = f"int{quant_config.activation_bits}"
    global_config = cmt_coreml.OpLinearQuantizerConfig(
        mode="linear",
        dtype=dtype,
        granularity=cmt_config.CompressionGranularity.PER_TENSOR,
    )
    optimization_config = cmt_coreml.OptimizationConfig(global_config=global_config)
    logger.info(
        "Applying activation quantization (%s) with %d calibration samples",
        dtype,
        len(calibration_data),
    )
    return cmt_coreml.linear_quantize_activations(
        mlmodel,
        optimization_config,
        sample_data=list(calibration_data),
    )


def quantization_metadata(quant_config: QuantizationConfig) -> Dict[str, str]:
    """Return metadata entries describing the quantization setup."""

    return quant_config.metadata_entries()


__all__ = [
    "QuantizationConfig",
    "QuantizationMode",
    "apply_post_training_quantization",
    "collect_coreml_calibration_samples",
    "collect_torch_calibration_samples",
    "prepare_torch_model_for_quantization",
    "quantization_metadata",
    "resolve_quantization_config",
]
