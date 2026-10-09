---
title: Model Tables
---
# Model Tables

```{warning}
`HookedTransformer` is deprecated as of TransformerLens 3.0 and will be removed in the next major version. New code should use [`TransformerBridge`](migrating_to_v3.md) instead. Existing `HookedTransformer` code continues to work through the 3.x branch via a compatibility layer. See the [migration guide](migrating_to_v3.md) for conversion recipes.
```

TransformerLens 3.0 provides two model loading paths, each with its own set of supported models.

- **HookedTransformer** -- The original TransformerLens models with full hook-point access and mechanistic interpretability support.
- **TransformerBridge Models** -- Automatic compatibility layer for thousands of HuggingFace models across supported architectures.

## Reading support status

Having an adapter is not the same as being verified. Three states apply to a TransformerBridge model family:

- **Implemented and verified** — an adapter exists and the model shows status "verified" in the TransformerBridge table, meaning `verify_models` recorded logit parity with HuggingFace.
- **Implemented, unverified** — an adapter exists, but no model of the family (or only a tiny random one) has a verified entry. Treat results as untested.
- **Unsupported** — no adapter; the architecture cannot be loaded.

Families that are currently **implemented but unverified** on real checkpoints include Cohere Command-R / Command-R+ / Aya, DeepSeek V2, DeepSeek V3, and DeepSeek-R1 (which loads through the DeepSeek V3 adapter). **Unsupported** relatives include Cohere Command R7B and Command A (`Cohere2ForCausalLM`) and DeepSeek V3.2 and V4. Always check the per-model status in the table rather than assuming a family name implies support.

```{toctree}
/generated/model_properties_table
/generated/transformer_bridge_models
```
