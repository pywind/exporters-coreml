<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# 🤗 Exporters – Core ML for LLMs

👷 **WORK IN PROGRESS** 👷

This repository focuses on exporting PyTorch-based large language models from 🤗 Transformers to Core ML. The exported models target Apple's ML Program format and are ready to run on Apple Silicon CPUs, GPUs and, when possible, the Apple Neural Engine (ANE).

## What's included?

* Ready-made Core ML configurations for popular decoder-only architectures such as **Llama**, **Mistral**, **Qwen/Qwen2**, **Phi-3**, and **GPT-2**.
* Support for **key/value cache** inputs and outputs to unlock fast autoregressive decoding.
* Built-in selection of Core ML **compute units**, allowing exports optimised for ANE acceleration (`ComputeUnit.ALL` or `ComputeUnit.CPU_AND_NE`).
* Simplified configuration classes that mirror the latest 🤗 Transformers APIs.

## Installation

Clone the repository and install it as an editable package:

```bash
$ git clone https://github.com/huggingface/exporters.git
$ cd exporters
$ pip install -e .
```

The exporter requires Python 3.9+, [PyTorch](https://pytorch.org) and the latest versions of [transformers](https://github.com/huggingface/transformers) and [coremltools](https://github.com/apple/coremltools).

## Exporting a model

The `exporters.coreml` package can be used from the command line. The example below exports a quantised Llama checkpoint with key/value caches enabled:

```bash
python -m exporters.coreml \
  --model meta-llama/Llama-2-7b-hf \
  --feature text-generation-with-past \
  --quantize float16 \
  --compute_units cpu_and_ne \
  --use_past \
  exported/
```

The command downloads the PyTorch checkpoint, traces it with TorchScript, and converts the traced module to Core ML. The resulting package is saved as `exported/Model.mlpackage` unless a different filename is provided. If the conversion runs on macOS 12 or later, a validation step compares the Core ML outputs to the original PyTorch model.

### Feature selection

Only causal language modelling features are currently supported:

| Feature                    | Description                                         |
|---------------------------|-----------------------------------------------------|
| `text-generation`         | Export without cache tensors.                       |
| `text-generation-with-past` | Export with key/value cache inputs and outputs.  |

Synonym task names such as `causal-lm` and `causal-lm-with-past` continue to work for backwards compatibility.

### Supported architectures

The exporter ships Core ML configuration helpers for the following model types:

* `llama`
* `mistral`
* `qwen` / `qwen2`
* `phi3`
* `gpt2`

Each configuration automatically reads the number of decoder layers, attention heads and context length from the Transformers config object.

## Programmatic API

You can also interact with the exporter in Python:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from exporters.coreml import CoreMLConfig, export

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torchscript=True)

coreml_config = CoreMLConfig.with_past(model.config, task="text-generation")
mlmodel = export(tokenizer, model, coreml_config, quantize="float16")
mlmodel.save("Llama.mlpackage")
```

The resulting `MLModel` contains flexible sequence lengths for logits and cache tensors, making it suitable for iterative generation on device.

## Validation

The helper `validate_model_outputs` compares the Core ML model against the reference PyTorch module using randomly generated inputs. Validation requires macOS 12+ because it loads the exported model through the Core ML framework.

```python
from exporters.coreml.validate import validate_model_outputs
validate_model_outputs(coreml_config, tokenizer, model, mlmodel, atol=1e-4)
```

## Troubleshooting

* Ensure you are using the latest `coremltools` release – decoder-only models rely on ML Program features introduced in Core ML 6.
* When tracing extremely large models, monitor GPU memory usage. You might need to set `torch.backends.cuda.matmul.allow_tf32 = False` or cast weights to `float16` ahead of time.
* The exporter only supports PyTorch checkpoints. TensorFlow or Flax models should be converted to PyTorch first.

## License

The code in this repository is released under the Apache 2.0 license. Consult the [LICENSE](LICENSE) file for the full text.
