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
"""Feature registry for exporting causal language models to Core ML."""

from __future__ import annotations

from functools import partial, reduce
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Type, Union

try:  # pragma: no cover - optional dependency
    from transformers import PretrainedConfig, is_torch_available
    from transformers.models.auto import AutoModelForCausalLM
except ImportError:  # pragma: no cover
    PretrainedConfig = Any

    def is_torch_available() -> bool:
        return False

    AutoModelForCausalLM = None

from .config import CoreMLConfig
from ..utils import logging


if TYPE_CHECKING:  # pragma: no cover - type checking only
    from transformers import PreTrainedModel


logger = logging.get_logger(__name__)


def supported_features_mapping(
    *supported_features: str, coreml_config_cls: str
) -> Dict[str, Callable[[PretrainedConfig], CoreMLConfig]]:
    """Generate a mapping from feature names to Core ML configuration constructors."""

    if coreml_config_cls is None:
        raise ValueError("A CoreMLConfig class must be provided")

    import exporters.coreml.models

    config_cls: Any = exporters.coreml
    for attr_name in coreml_config_cls.split("."):
        if hasattr(config_cls, attr_name):
            config_cls = getattr(config_cls, attr_name)

    mapping: Dict[str, Callable[[PretrainedConfig], CoreMLConfig]] = {}
    for feature in supported_features:
        if feature.endswith("-with-past"):
            task = feature.replace("-with-past", "")
            mapping[feature] = partial(config_cls.with_past, task=task)
        else:
            mapping[feature] = partial(config_cls.from_model_config, task=feature)
    return mapping


class FeaturesManager:
    """Utility class centralising model/task discovery for the Core ML exporter."""

    _TASKS_TO_AUTOMODELS: Dict[str, Type] = {}
    if is_torch_available():  # pragma: no cover - executed when transformers+torch are installed
        _TASKS_TO_AUTOMODELS = {
            "text-generation": AutoModelForCausalLM,
            "text-generation-with-past": AutoModelForCausalLM,
        }

    _SYNONYM_TASK_MAP = {
        "causal-lm": "text-generation",
        "causal-lm-with-past": "text-generation-with-past",
        "default": "text-generation",
        "default-with-past": "text-generation-with-past",
    }

    _SUPPORTED_MODEL_TYPE = {
        "gpt2": supported_features_mapping(
            "text-generation",
            "text-generation-with-past",
            coreml_config_cls="models.GPT2CoreMLConfig",
        ),
        "llama": supported_features_mapping(
            "text-generation",
            "text-generation-with-past",
            coreml_config_cls="models.LlamaCoreMLConfig",
        ),
        "mistral": supported_features_mapping(
            "text-generation",
            "text-generation-with-past",
            coreml_config_cls="models.MistralCoreMLConfig",
        ),
        "phi3": supported_features_mapping(
            "text-generation",
            "text-generation-with-past",
            coreml_config_cls="models.Phi3CoreMLConfig",
        ),
        "qwen": supported_features_mapping(
            "text-generation",
            "text-generation-with-past",
            coreml_config_cls="models.QwenCoreMLConfig",
        ),
        "qwen2": supported_features_mapping(
            "text-generation",
            "text-generation-with-past",
            coreml_config_cls="models.Qwen2CoreMLConfig",
        ),
    }

    AVAILABLE_FEATURES = sorted(reduce(lambda s1, s2: s1 | s2, (v.keys() for v in _SUPPORTED_MODEL_TYPE.values())))
    AVAILABLE_FEATURES_INCLUDING_LEGACY = AVAILABLE_FEATURES + list(_SYNONYM_TASK_MAP.keys())

    @staticmethod
    def get_supported_features_for_model_type(
        model_type: str, model_name: Optional[str] = None
    ) -> Dict[str, Callable[[PretrainedConfig], CoreMLConfig]]:
        model_type = model_type.lower()
        if model_type not in FeaturesManager._SUPPORTED_MODEL_TYPE:
            model_type_and_model_name = f"{model_type} ({model_name})" if model_name else model_type
            raise KeyError(
                f"{model_type_and_model_name} is not supported. Supported types:"
                f" {sorted(FeaturesManager._SUPPORTED_MODEL_TYPE.keys())}"
            )
        return FeaturesManager._SUPPORTED_MODEL_TYPE[model_type]

    @staticmethod
    def feature_to_task(feature: str) -> str:
        return feature.replace("-with-past", "")

    @staticmethod
    def map_from_synonym(feature: str) -> str:
        return FeaturesManager._SYNONYM_TASK_MAP.get(feature, feature)

    @staticmethod
    def _validate_framework_choice(framework: str) -> None:
        if framework != "pt":
            raise ValueError("The Core ML exporter currently only supports PyTorch models")
        if not is_torch_available():
            raise RuntimeError("Cannot export models because PyTorch is not installed")

    @staticmethod
    def get_model_class_for_feature(feature: str, framework: str = "pt") -> Type:
        task = FeaturesManager.feature_to_task(feature)
        FeaturesManager._validate_framework_choice(framework)
        if task not in FeaturesManager._TASKS_TO_AUTOMODELS:
            raise KeyError(f"Unknown task: {feature}")
        return FeaturesManager._TASKS_TO_AUTOMODELS[task]

    @staticmethod
    def get_model_from_feature(
        feature: str, model: str, framework: str = "pt", cache_dir: Optional[str] = None
    ) -> "PreTrainedModel":
        model_class = FeaturesManager.get_model_class_for_feature(feature, framework)
        return model_class.from_pretrained(model, cache_dir=cache_dir, torchscript=True)

    @staticmethod
    def check_supported_model_or_raise(
        model: "PreTrainedModel", feature: str = "text-generation"
    ) -> Tuple[str, Callable[[PretrainedConfig], CoreMLConfig]]:
        model_type = model.config.model_type.replace("-", "_")
        model_name = getattr(model, "name", "")
        model_features = FeaturesManager.get_supported_features_for_model_type(model_type, model_name=model_name)
        if feature not in model_features:
            raise ValueError(
                f"{model.config.model_type} doesn't support feature {feature}. Supported values are:"
                f" {sorted(model_features.keys())}"
            )
        return model.config.model_type, model_features[feature]

    @staticmethod
    def get_config(model_type: str, feature: str) -> Callable[[PretrainedConfig], CoreMLConfig]:
        return FeaturesManager._SUPPORTED_MODEL_TYPE[model_type][feature]


__all__ = ["FeaturesManager", "supported_features_mapping"]
