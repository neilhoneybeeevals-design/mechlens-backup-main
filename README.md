# mechlens

Personal mechanistic-interpretability toolkit.

A `HookedTransformer` wrapper for GPT-style language models that exposes every internal activation
(residual stream, attention heads, MLPs) via named hooks. Loads 9,000+ open-source models across 50+
architecture families and lets you cache, edit, remove, or replace activations as the model runs.

## Install

```bash
pip install -e .
```

## Quickstart

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2")
logits, cache = model.run_with_cache("The capital of France is")
print(cache["blocks.0.attn.hook_pattern"].shape)
```

See `demos/Main_Demo.ipynb` for the full walkthrough.

## Docs

Docs source lives in `docs/`. Build locally with:

```bash
cd docs && make html
```

## License

MIT. See [LICENSE](LICENSE).
