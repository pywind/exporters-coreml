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
"""Command line interface for exporting causal language models to Core ML."""

from __future__ import annotations

import warnings
from argparse import ArgumentParser
from pathlib import Path

from coremltools import ComputeUnit
from coremltools.models import MLModel
from coremltools.models.utils import _is_macos, _macos_version

from transformers.models.auto import AutoProcessor, AutoTokenizer
from transformers.onnx.utils import get_preprocessor

from .convert import export
from .features import FeaturesManager
from .validate import validate_model_outputs
from .quantization import resolve_quantization_config
from ..utils import logging


logger = logging.get_logger(__name__)


def convert_model(preprocessor, model, model_coreml_config, args, use_past: bool = False):
    coreml_config = model_coreml_config(model.config, use_past=use_past)

    compute_units = ComputeUnit.ALL
    if args.compute_units == "cpu_and_gpu":
        compute_units = ComputeUnit.CPU_AND_GPU
    elif args.compute_units == "cpu_only":
        compute_units = ComputeUnit.CPU_ONLY
    elif args.compute_units == "cpu_and_ne":
        compute_units = ComputeUnit.CPU_AND_NE

    quant_config = resolve_quantization_config(args.quantize)
    quant_config.calibration_samples = args.calibration_limit
    quant_config.calibration_prompts = args.calibration_prompts
    quant_config.gptq_block_size = args.gptq_block_size
    if args.qat_checkpoint is not None:
        quant_config.qat_checkpoint = args.qat_checkpoint

    mlmodel = export(
        preprocessor,
        model,
        coreml_config,
        quantize=quant_config,
        compute_units=compute_units,
    )

    filename = args.output
    mlmodel.save(filename.as_posix())

    if args.atol is None:
        args.atol = coreml_config.atol_for_validation

    if not _is_macos() or _macos_version() < (12, 0):
        logger.info("Skipping model validation, requires macOS 12.0 or later")
    else:
        mlmodel_cpu = MLModel(filename.as_posix(), compute_units=ComputeUnit.CPU_ONLY)
        validate_model_outputs(coreml_config, preprocessor, model, mlmodel_cpu, args.atol)

    logger.info(f"All good, model saved at: {filename}")


def main():
    parser = ArgumentParser("Hugging Face Transformers Core ML exporter for LLMs")
    parser.add_argument("-m", "--model", type=str, required=True, help="Model identifier on the Hugging Face Hub or local path")
    parser.add_argument(
        "--feature",
        choices=list(FeaturesManager.AVAILABLE_FEATURES_INCLUDING_LEGACY),
        default="text-generation",
        help="The model feature to export.",
    )
    parser.add_argument("--atol", type=float, default=None, help="Absolute tolerance used during validation")
    parser.add_argument(
        "--use_past",
        action="store_true",
        help="Export the model with key/value caches for fast autoregressive decoding.",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="float16",
        help=(
            "Quantization strategy. Supported values include float32, float16, "
            "rtn-int4, rtn-int8, activation-int8, gptq-int4 and qat-int8."
        ),
    )
    parser.add_argument(
        "--compute_units",
        type=str,
        choices=["all", "cpu_and_gpu", "cpu_only", "cpu_and_ne"],
        default="all",
        help="Hardware units to optimise for during conversion.",
    )
    parser.add_argument(
        "--calibration_prompts",
        type=Path,
        default=None,
        help="Optional text file containing one prompt per line for calibration-aware quantization.",
    )
    parser.add_argument(
        "--calibration_limit",
        type=int,
        default=128,
        help="Maximum number of samples to draw when building calibration datasets.",
    )
    parser.add_argument(
        "--gptq_block_size",
        type=int,
        default=128,
        help="Block size to use for GPTQ quantization (only applicable to gptq-* modes).",
    )
    parser.add_argument(
        "--qat_checkpoint",
        type=Path,
        default=None,
        help="Path to a checkpoint produced via quantization-aware training (used for qat-* modes).",
    )
    parser.add_argument(
        "--preprocessor",
        type=str,
        choices=["auto", "tokenizer", "processor"],
        default="auto",
        help="Type of preprocessor to instantiate for dummy inputs.",
    )
    parser.add_argument("output", type=Path, help="Destination path for the Core ML package or model file")

    args = parser.parse_args()

    if (not args.output.is_file()) and (args.output.suffix not in [".mlpackage", ".mlmodel"]):
        args.output = args.output.joinpath("Model.mlpackage")
    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True)

    if args.preprocessor == "auto":
        preprocessor = get_preprocessor(args.model)
    elif args.preprocessor == "tokenizer":
        preprocessor = AutoTokenizer.from_pretrained(args.model)
    elif args.preprocessor == "processor":
        preprocessor = AutoProcessor.from_pretrained(args.model)
    else:  # pragma: no cover
        raise ValueError(f"Unknown preprocessor type '{args.preprocessor}'")

    feature = args.feature
    args.feature = FeaturesManager.map_from_synonym(feature)
    if feature != args.feature:
        warnings.warn(f"Feature '{feature}' is deprecated, please use '{args.feature}' instead.", FutureWarning)

    model = FeaturesManager.get_model_from_feature(args.feature, args.model, framework="pt")
    model_kind, model_coreml_config = FeaturesManager.check_supported_model_or_raise(model, feature=args.feature)

    convert_model(preprocessor, model, model_coreml_config, args, use_past=args.use_past)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    main()
