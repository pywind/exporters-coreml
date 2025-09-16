# Supported Core ML models

The simplified exporter focuses on decoder-only large language models implemented in PyTorch. The following model types are covered by ready-made configuration classes:

| Model type | Notes |
|------------|-------|
| `llama`    | Works with the latest Meta Llama checkpoints. |
| `mistral`  | Includes Mixtral-like architectures sharing the same configuration schema. |
| `qwen` / `qwen2` | Covers the original Qwen series and the updated Qwen2 family. |
| `phi3`     | Supports Microsoft's Phi-3 models. |
| `gpt2`     | Provides a lightweight option for testing the exporter pipeline. |

Other architectures can be supported by subclassing `CoreMLConfig` and registering the subclass in `FeaturesManager._SUPPORTED_MODEL_TYPE`.
