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
"""Validation utilities for the Core ML exporter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Tuple, Union

import coremltools as ct
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
from ..utils import logging


if TYPE_CHECKING:  # pragma: no cover - type checking only
    from transformers.feature_extraction_utils import FeatureExtractionMixin
    from transformers.processing_utils import ProcessorMixin
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.modeling_utils import PreTrainedModel


logger = logging.get_logger(__name__)


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def validate_model_outputs(
    config: CoreMLConfig,
    preprocessor: Union["PreTrainedTokenizer", "FeatureExtractionMixin", "ProcessorMixin"],
    reference_model: "PreTrainedModel",
    mlmodel: ct.models.MLModel,
    atol: float,
) -> None:
    if not is_torch_available():  # pragma: no cover
        raise RuntimeError("Validation requires PyTorch to be installed")

    from transformers.modeling_utils import PreTrainedModel as HFPreTrainedModel

    if not isinstance(reference_model, HFPreTrainedModel):  # pragma: no cover
        raise ValueError("Validation expects a PyTorch PreTrainedModel")

    logger.info("Validating Core ML model...")

    dummy_inputs = config.generate_dummy_inputs(preprocessor, framework=TensorType.PYTORCH)

    reference_inputs: Dict[str, object] = {}
    past_key_values: List[List[object]] = []
    coreml_inputs: Dict[str, object] = {}

    for name, desc in config.inputs.items():
        reference_value, coreml_value = dummy_inputs[name]
        if name.startswith("past_key_values"):
            layer_index = int(name.split("_")[3])
            while len(past_key_values) <= layer_index:
                past_key_values.append([])
            past_key_values[layer_index].append(reference_value)
        else:
            reference_inputs[name] = reference_value
        coreml_inputs[desc.name] = coreml_value

    if past_key_values:
        reference_inputs["past_key_values"] = [tuple(layer) for layer in past_key_values]

    import torch

    reference_model = reference_model.to("cpu").eval()
    with torch.no_grad():
        reference_outputs = reference_model(**reference_inputs, return_dict=True)

    if "past_key_values" in reference_outputs:
        for idx, (key, value) in enumerate(reference_outputs["past_key_values"]):
            reference_outputs[f"present_{idx}_key"] = key
            reference_outputs[f"present_{idx}_value"] = value

    coreml_outputs = mlmodel.predict(coreml_inputs)

    for name, desc in config.outputs.items():
        if desc.name not in coreml_outputs:
            logger.warning(f"Core ML output '{desc.name}' missing from prediction results")
            continue

        coreml_value = _numpy(coreml_outputs[desc.name])
        reference_value = _numpy(reference_outputs[name])

        if coreml_value.shape != reference_value.shape:
            raise ValueError(
                f"Output '{name}' shape mismatch: Core ML {coreml_value.shape} vs reference {reference_value.shape}"
            )

        if not np.allclose(coreml_value, reference_value, atol=atol):
            max_diff = np.max(np.abs(coreml_value - reference_value))
            raise ValueError(f"Output '{name}' differs more than allowed tolerance (max diff={max_diff})")

    logger.info("All good, Core ML model outputs match the reference model")


__all__ = ["validate_model_outputs"]
