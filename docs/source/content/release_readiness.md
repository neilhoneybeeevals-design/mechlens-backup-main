---
orphan: true
---

# Release Readiness — Command-R (Cohere) Model Support

**Status: BLOCKED / NO-GO.** Last updated 2026-10-09. Branch `fix/cohere-rope-qknorm-logit-scale`, code commit `1b7c4a9` (base `main` @ `a4c3648`).

The Command-R fixes below have **not been executed**. No test in this repository has been run against them. Do not release Command-R support, or anything depending on the shared attention bridge changes, until the [exit criteria](#exit-criteria) are met.

Everything on this page is labelled as one of:

- **Verified** — backed by command output or by source that was read directly.
- **Not run** — attempted or planned, but no result exists.
- **Unresolved** — a question nobody has answered yet.

## 1. Fixes on this branch

All three defects have the same cause: the Cohere adapter was written as if `PositionEmbeddingsAttentionBridge` delegated to `CohereAttention.forward`. It does not; it reimplements attention. The defects were found by reading this repository against the pinned `transformers` 5.8.1 source. They have not been reproduced at runtime.

| # | Defect | Fix |
|---|---|---|
| 1 | The bridge always applied Llama's half-split RoPE rotation. HF Cohere rotates adjacent pairs. The two agree only when `d_head == 2`. | `rotary_interleaved` flag on the attention bridge; the Cohere adapter sets it. Default behaviour for other architectures is unchanged. |
| 2 | Checkpoints with `use_qk_norm=True` (Command-R+) have `q_norm` / `k_norm` that the adapter never declared, so the bridge raises at boot. The existing phase logic would also have applied a `[n_heads, d_head]` norm weight after the head/seq transpose. | Adapter declares both norms when `use_qk_norm` is set; new `per_head` phase applies them on `[B, S, H, D]`. |
| 3 | `process_weights` folds `logit_scale` into `unembed.weight`, but the bridge forward still runs `CohereForCausalLM.forward`, which multiplies by `logit_scale` again. | New `ArchitectureAdapter.finalize_weight_processing` hook, called at the end of `TransformerBridge.process_weights`; Cohere resets `hf_model.logit_scale` to `1.0` once the fold is written. |

Registry evidence consistent with defect 1 (recorded before this branch, in `supported_models.json`): `trl-internal-testing/tiny-CohereForCausalLM` (`d_head = 2`) is the only verified Cohere entry; `hyper-accel/tiny-random-cohere` (`d_head = 128`) is `FAILED` with `P1=50.0% (failed: forward_pass_logits)`.

### Files changed in `1b7c4a9` (9 files, +453 / −61)

| File | Change |
|---|---|
| `transformer_lens/model_bridge/supported_architectures/cohere.py` | Interleaved RoPE, QK-norm submodules, single-application `logit_scale` |
| `transformer_lens/model_bridge/generalized_components/position_embeddings_attention.py` | `rotary_interleaved`, `_apply_rotary`, `per_head` QK-norm phase |
| `transformer_lens/model_bridge/architecture_adapter.py` | `finalize_weight_processing` no-op base hook |
| `transformer_lens/model_bridge/bridge.py` | Calls the hook at the end of `process_weights` |
| `transformer_lens/model_bridge/sources/transformers.py` | Maps HF `use_qk_norm` onto the declared config field |
| `transformer_lens/benchmarks/component_outputs.py` | Per-head QK norms get a `[B, S, H, d_head]` test input |
| `transformer_lens/model_bridge/supported_architectures/AGENTS.md` | Guidance on reimplemented attention and degenerate parity models |
| `tests/unit/model_bridge/supported_architectures/test_cohere_adapter.py` | Rotation, QK-norm, and `logit_scale` unit tests |
| `tests/integration/model_bridge/test_cohere_adapter.py` | Offline parity tests (`d_head = 8`, GQA, with and without QK-norm) |

## 2. Checks that ran

Run on 2026-10-09 in a sandbox with no access to PyPI or the Hugging Face Hub. `torch` and `transformers` were not installed. The tools came from a local cache and are **not the versions this repo pins**, so these results are indicative only.

| Check | Result |
|---|---|
| `git status`, `git ls-remote` | Verified: working tree clean; local and remote branch both at `1b7c4a9`. |
| `ast.parse` on the 8 changed `.py` files | Verified: all parse. |
| `black --check` on the 8 files (black 26.5.1; pinned 23.12.1) | 5 unchanged, 3 "would reformat". The same 3 files are flagged on `main`; no reported hunk touches a changed line. |
| `ruff check --select F,I` on the 8 files (ruff 0.16.8; repo uses pycln + isort 5.8.0) | 10 findings. The identical 10 exist on `main`; none are new. |
| `mypy --follow-imports=silent` on the 6 changed source files (mypy 2.3.1; pinned 1.17.0; torch absent) | 1 error in `component_outputs.py` (`TransformerBridge` attribute). The same error exists on `main`. |
| `# type: ignore` count across the 8 files | 33 on the branch, 36 on `main`. |
| Plain-Python re-derivation of both RoPE conventions (scratch script, not repo code) | Max difference `0.000000` at `d_head = 2`, `2.615650` at 4, `1.382113` at 8. Confirms the arithmetic behind defect 1, not the implementation. |
| Static review of the diff by a second reviewer | Found two problems, both fixed before `1b7c4a9`. No runtime evidence. |

## 3. Tests blocked

| Command | Outcome |
|---|---|
| `uv sync` | **Failed**: `403 Forbidden` from `files.pythonhosted.org` on a torch dependency. No environment was created. |
| `pytest` on both Cohere test files | **Zero tests ran**: `tests/conftest.py:15: ModuleNotFoundError: No module named 'torch'` (exit 4). |

Not run as a consequence:

- Every new test, including the offline parity tests that need no Hub access.
- The existing Cohere tests that load `trl-internal-testing/tiny-CohereForCausalLM` (need the Hub).
- Tests for every other adapter that uses `PositionEmbeddingsAttentionBridge`.
- `make unit-test`, `make integration-test`, `make acceptance-test`, `make docstring-test`, `make test-pr`.
- `make check-format` and `uv run mypy .` with the pinned tool versions.
- `verify_models` for `CohereForCausalLM`.

## 4. Remaining risks

**Command-R**

- The fixes are unproven. The new tests may fail for reasons the static review missed, including tolerance choices.
- No real Command-R, Command-R+, or Aya checkpoint has ever been verified here. Of 19 `CohereForCausalLM` registry entries, 17 are skipped (gated, over the memory limit, or quantized), 1 is verified with issues, and 1 is failed.
- QK-norm together with full compatibility mode (`fold_ln`, centering) has no test.
- `hf_model.logit_scale` is mutated in place by `process_weights`. Building a second bridge on the same already-processed HF model would fold the scale again. Unresolved whether any workflow does this.
- Registry statuses are stale against this branch and may only be changed by `verify_models`.

**Shared adapter regressions**

- `position_embeddings_attention.py` is used by many architectures. The view/transpose block was restructured and RoPE now goes through `_apply_rotary`. By reading, the default path is equivalent; this is not confirmed by any test.
- A `[n_heads, d_head]` QK-norm weight now selects `per_head` instead of `post_reshape`. Unresolved whether any other adapter, in-tree or out-of-tree, wraps a norm with a 2-D weight; this was not checked exhaustively.
- `TransformerBridge.process_weights` now calls `finalize_weight_processing` for every adapter. The base implementation is a no-op.
- `get_random_inputs` in the attention bridge falls back to an identity rotation (`cos = 1`, `sin = 0`) when the rotary module is unset or raises. Verified in source. Component-level benchmarks on that path cannot detect a RoPE convention error; only full-model parity can.

## 5. Upstream #1126 and #1127

Both numbers refer to **merged pull requests** in `TransformerLensOrg/TransformerLens`, not issues. This repository (`mechlens-backup-main`) has no issues or pull requests with these numbers.

### Verified facts (from the upstream PR pages and diffs)

| | #1126 "Attention cleanup" | #1127 "Benchmarking cross comparison revision" |
|---|---|---|
| State | Merged 2025-11-12 into `dev-3.x-folding` (merge commit `4715592`) | Merged 2025-11-12 into `dev-3.x-folding` (merge commit `eb60dea`) |
| Description / review | Unedited template; no linked issue; no reviews | Unedited template; no linked issue; no reviews |
| CI | "12 of 13 checks passed" | "12 of 13 checks passed" |
| Relevant changes | Renamed `gemma3_attention.py` to `position_embeddings_attention.py`, turning a Gemma-specific attention bridge into the shared one; added the identity-rotation fallback; changed `set_processed_weights` to dict form across components | Added a `cross_model` benchmark mode that checks shapes and skips value comparison; `NormalizationBridge` reads `eps` / `variance_epsilon` from the HF module; `position_embeddings_attention.py` forced `output_attentions=True` and fired an approximate `hook_attn_scores` |
| Cohere / `logit_scale` | Not mentioned anywhere in the diff | Not mentioned anywhere in the diff |

Also verified in this repository:

- The identity-rotation fallback from the #1126 lineage is still present in `get_random_inputs`.
- The string `cross_model` no longer appears anywhere under `transformer_lens/`.
- Neither PR touched the Cohere adapter. `cohere.py` does not appear in either diff.

### Impact assessment

- **#1126** is the origin of the shared attention bridge that defects 1 and 2 live in. It is relevant as context: the bridge began as Gemma-3 attention, and Gemma and Llama share the half-split RoPE convention. It did not introduce the Cohere defects by itself, because Cohere support did not exist in that diff.
- **#1127** shaped what the benchmarks compare. Its one identifiable link to this branch is norm epsilon handling: the native-forward branch of `NormalizationBridge`, which the new Cohere QK norms use, reads `variance_epsilon` from the HF module when computing hook values.

### Unresolved questions

- Whether this repository's history contains these PRs as merged. History here is a single squashed commit, so it cannot be traced; only the present code can be compared.
- Which CI check failed on each PR. The pages do not say.
- When the bridge changed from wrapping HF attention to reimplementing it, and whether RoPE convention was considered at that point. Neither PR diff shows `apply_rotary_pos_emb`.
- What replaced `cross_model`, and whether any current benchmark still skips value comparison in a way that could pass a wrong-RoPE model.
- Whether "#1126" and "#1127" were meant to refer to something other than these upstream PRs. If so, this section needs redoing against the right source.

## 6. Recommendation

**BLOCKED / NO-GO** for releasing Command-R support and for merging the shared attention bridge change into a release branch.

(exit-criteria)=
### Exit criteria

All of the following must be produced as actual command output on a machine with `torch` installed:

1. `uv sync` succeeds.
2. `uv run pytest tests/unit/model_bridge/supported_architectures/test_cohere_adapter.py -v` passes.
3. `uv run pytest tests/integration/model_bridge/test_cohere_adapter.py -v` passes, including the offline `d_head = 8` parity classes with and without QK-norm.
4. `make test-pr` passes, covering every adapter that shares the attention bridge.
5. `make check-format` and `uv run mypy .` pass with the pinned versions.
6. `verify_models --reverify --architectures CohereForCausalLM` passes Phase 1 at 100% on at least one checkpoint with `d_head > 2`, and on a `use_qk_norm=True` checkpoint before Command-R+ is advertised.

Until then, treat every claim in section 1 as a hypothesis supported by source reading only.
