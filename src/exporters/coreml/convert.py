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
"""Conversion helpers to export PyTorch decoder-only models to Core ML."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Mapping, Tuple, Union

import coremltools as ct
from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY
import numpy as np

try:  # pragma: no cover - optional dependency
    from transformers.utils import TensorType, is_torch_available
except ImportError:  # pragma: no cover
    from enum import Enum

    class TensorType(Enum):
        PYTORCH = "pt"
        TENSORFLOW = "tf"

    def is_torch_available() -> bool:
        return False

from .config import CoreMLConfig
from .quantization import (
    QuantizationConfig,
    apply_post_training_quantization,
    quantization_metadata,
    resolve_quantization_config,
)
from ..utils import logging


if TYPE_CHECKING:  # pragma: no cover - type checking only
    from transformers.feature_extraction_utils import FeatureExtractionMixin
    from transformers.processing_utils import ProcessorMixin
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.modeling_utils import PreTrainedModel


logger = logging.get_logger(__name__)


def _sequence_shape(config: CoreMLConfig, array: np.ndarray) -> ct.Shape:
    shape = list(array.shape)
    if len(shape) >= 2:
        shape[1] = ct.RangeDim(1, config.max_sequence_length)
    return ct.Shape(shape)


def get_input_types(
    config: CoreMLConfig,
    dummy_inputs: Mapping[str, Tuple[np.ndarray, np.ndarray]],
) -> List[ct.TensorType]:
    input_types: List[ct.TensorType] = []
    for name, input_desc in config.inputs.items():
        _, coreml_value = dummy_inputs[name]
        if name.startswith("past_key_values"):
            shape = list(coreml_value.shape)
            if len(shape) >= 3:
                shape[2] = ct.RangeDim(0, -1)
            input_types.append(ct.TensorType(name=input_desc.name, shape=ct.Shape(shape), dtype=np.float32))
        elif name in {"input_ids", "attention_mask"}:
            input_types.append(ct.TensorType(name=input_desc.name, shape=_sequence_shape(config, coreml_value), dtype=np.int32))
        else:
            input_types.append(ct.TensorType(name=input_desc.name, shape=coreml_value.shape))
    return input_types


if is_torch_available():  # pragma: no cover - executed when torch is installed
    import torch

    class Wrapper(torch.nn.Module):
        def __init__(self, model: "PreTrainedModel", config: CoreMLConfig) -> None:
            super().__init__()
            self.model = model.eval()
            self.config = config

        def forward(self, *model_inputs):
            tensor_inputs = list(model_inputs)
            inputs = tensor_inputs[0]
            attention_mask = None
            remaining = len(tensor_inputs)

            if self.config.uses_attention_mask and remaining >= 2:
                attention_mask = tensor_inputs[1]

            kwargs = {"return_dict": False, "input_ids": inputs}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask

            offset = 1 + (1 if attention_mask is not None else 0)

            if self.config.use_past:
                past_key_values = []
                for layer in range(self.config.num_layers):
                    key = tensor_inputs[offset + layer * 2]
                    value = tensor_inputs[offset + layer * 2 + 1]
                    past_key_values.append((key, value))
                kwargs["past_key_values"] = past_key_values
                kwargs["use_cache"] = True

            outputs = self.model(**kwargs)
            logits = outputs[0]

            if self.config.use_past:
                presents = outputs[1]
                flat_presents = tuple(component for layer_cache in presents for component in layer_cache)
                return (logits,) + flat_presents

            return (logits,)


def export_pytorch(
    preprocessor: Union["PreTrainedTokenizer", "FeatureExtractionMixin", "ProcessorMixin"],
    model: "PreTrainedModel",
    config: CoreMLConfig,
    quantize: Union[str, QuantizationConfig] = "float16",
    compute_units: ct.ComputeUnit = ct.ComputeUnit.ALL,
) -> ct.models.MLModel:
    if not is_torch_available():  # pragma: no cover
        raise RuntimeError("PyTorch needs to be installed to export models to Core ML")

    import torch
    from transformers.modeling_utils import PreTrainedModel as HFPreTrainedModel

    if not isinstance(model, HFPreTrainedModel):  # pragma: no cover
        raise ValueError(f"Unsupported model type: {type(model)}")

    logger.info(f"Using framework PyTorch: {torch.__version__}")

    if config.values_override is not None:
        for key, value in config.values_override.items():
            logger.info(f"Overriding {key} -> {value}")
            setattr(model.config, key, value)

    quant_config = resolve_quantization_config(quantize)

    dummy_inputs = config.generate_dummy_inputs(preprocessor, framework=TensorType.PYTORCH)

    example_inputs = [dummy_inputs[key][0] for key in config.inputs.keys()]

    wrapper = Wrapper(model, config).eval()
    with torch.no_grad():
        _ = wrapper(*example_inputs)

    traced_model = torch.jit.trace(wrapper, example_inputs, strict=False)

    with torch.no_grad():
        outputs = traced_model(*example_inputs)

    if isinstance(outputs, (tuple, list)):
        numpy_outputs = [output.detach().cpu().numpy() for output in outputs]
    else:  # pragma: no cover
        numpy_outputs = [outputs.detach().cpu().numpy()]

    convert_kwargs = {
        "compute_precision": quant_config.coreml_compute_precision,
    }

    input_tensors = get_input_types(config, dummy_inputs)

    patched_ops = config.patch_pytorch_ops()
    restore_ops = {}
    if patched_ops:
        for name, func in patched_ops.items():
            logger.info(f"Patching PyTorch op '{name}'")
            if name in _TORCH_OPS_REGISTRY:
                restore_ops[name] = _TORCH_OPS_REGISTRY[name]
                del _TORCH_OPS_REGISTRY[name]
            _TORCH_OPS_REGISTRY[name] = func

    mlmodel = ct.convert(
        traced_model,
        inputs=input_tensors,
        convert_to="mlprogram",
        compute_units=compute_units,
        **convert_kwargs,
    )

    mlmodel = apply_post_training_quantization(mlmodel, quant_config)

    for name, func in restore_ops.items():  # pragma: no cover
        if func is not None:
            _TORCH_OPS_REGISTRY[name] = func

    spec = mlmodel._spec
    output_descs = config.outputs

    for index, (key, output_desc) in enumerate(output_descs.items()):
        if index < len(numpy_outputs):
            ct.utils.rename_feature(spec, spec.description.output[index].name, output_desc.name, rename_inputs=False)
            mlmodel.output_description[output_desc.name] = output_desc.description

    user_metadata = quantization_metadata(quant_config)
    user_metadata["co.huggingface.exporters.task"] = config.task
    if hasattr(model.config, "architectures") and model.config.architectures:
        user_metadata["co.huggingface.exporters.architecture"] = model.config.architectures[0]
    if getattr(model.config, "transformers_version", None):
        user_metadata["transformers_version"] = model.config.transformers_version

    if user_metadata:
        spec.description.metadata.userDefined.update(user_metadata)
    spec.description.metadata.shortDescription = config.short_description

    mlmodel = ct.models.MLModel(spec, weights_dir=mlmodel.weights_dir)
    return mlmodel


def export(
    preprocessor: Union["PreTrainedTokenizer", "FeatureExtractionMixin", "ProcessorMixin"],
    model: "PreTrainedModel",
    config: CoreMLConfig,
    quantize: Union[str, QuantizationConfig] = "float16",
    compute_units: ct.ComputeUnit = ct.ComputeUnit.ALL,
) -> ct.models.MLModel:
    if not is_torch_available():  # pragma: no cover
        raise RuntimeError("PyTorch needs to be installed to export models to Core ML")

    return export_pytorch(preprocessor, model, config, quantize=quantize, compute_units=compute_units)


__all__ = ["export", "export_pytorch"]
