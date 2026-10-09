---
orphan: true
---

# Release Readiness — Command-R (Cohere) Model Support

**Status: BLOCKED / NO-GO.** Last updated 2026-10-09. The ordered list of checks, owners and pass criteria is in [section 9](#fastest-path). Branch `fix/cohere-rope-qknorm-logit-scale` (base `main` @ `a4c3648`). Code changes are in `1b7c4a9`, `68dbd36`, `1008f60` and `56ddd53`; the other commits are documentation.

The Command-R fixes below have **not been executed**. No test in this repository has been run against them. Do not release Command-R support, or anything depending on the shared attention bridge changes, until the [exit criteria](#exit-criteria) are met.

Everything on this page is labelled as one of:

- **Verified** — backed by command output or by source that was read directly.
- **Not run** — attempted or planned, but no result exists.
- **Unresolved** — a question nobody has answered yet.

## 1. Fixes on this branch

All three defects have the same cause: the Cohere adapter was written as if `PositionEmbeddingsAttentionBridge` delegated to `CohereAttention.forward`. It does not; it reimplements attention. The defects were found by reading this repository against the pinned `transformers` 5.8.1 source. They have not been reproduced at runtime.

| # | Defect | Fix |
|---|---|---|
| 1 | The bridge always applied Llama's half-split RoPE rotation. HF Cohere rotates adjacent pairs. The two agree only when `d_head == 2`. | `rotary_interleaved_cos_sin` config flag (default `False`) read by the attention bridge; the Cohere adapter sets it. Same design and helper as upstream PR #1873 (see section 7E). Default behaviour for other architectures is unchanged. |
| 2 | Checkpoints with `use_qk_norm=True` (Command-R+) have `q_norm` / `k_norm` that the adapter never declared, so the bridge raises at boot. The existing phase logic would also have applied a `[n_heads, d_head]` norm weight after the head/seq transpose. | Adapter declares both norms when `use_qk_norm` is set; new `per_head` phase applies them on `[B, S, H, D]`. |
| 3 | `process_weights` folds `logit_scale` into `unembed.weight`, but the bridge forward still runs `CohereForCausalLM.forward`, which multiplies by `logit_scale` again. | New `ArchitectureAdapter.finalize_weight_processing` hook, called at the end of `TransformerBridge.process_weights`; Cohere resets `hf_model.logit_scale` to `1.0` once the fold is written. |

Registry evidence consistent with defect 1 (recorded before this branch, in `supported_models.json`): `trl-internal-testing/tiny-CohereForCausalLM` (`d_head = 2`) is the only verified Cohere entry; `hyper-accel/tiny-random-cohere` (`d_head = 128`) is `FAILED` with `P1=50.0% (failed: forward_pass_logits)`.

Evidence on defect 3 (investigated 2026-10-09; nothing executed):

- **Code path, verified by reading.** `TransformerBridge.forward` has one route to logits, `self.original_model(...)`. `GeneralizedComponent.set_processed_weights` replaces `lm_head.weight` with the processed tensor, so after the fold the HF forward computes `lm_head_scaled(x) * self.logit_scale`. No other code reads or compensates for `logit_scale`.
- **Registry, historical.** `hyper-accel/tiny-random-cohere` failed a *required* Phase 3 test. Phase 3 compares the processed bridge with the same bridge's own unprocessed logits, so that failure is independent of the RoPE defect and shows weight processing changing the output on a non-degenerate Cohere model. It is consistent with a double-applied scale but does not isolate it: other processing steps could also contribute.
- **Why the degenerate model passed.** Phase 3 compares `log_softmax` with an absolute tolerance of 0.01. On `trl-internal-testing/tiny-CohereForCausalLM` the logits are far smaller than that, so a scale error cannot show.
- **Still uncertain.** Whether the fix yields exact parity has not been observed. `TestCohereOfflineProcessedLogitScale` decides it without Hub access: `test_scale_lives_in_exactly_one_place` checks the wiring without a forward pass, and `test_processed_forward_matches_hf` checks the logits.

### Files changed in `1b7c4a9` (9 files, +453 / −61)

| File | Change |
|---|---|
| `transformer_lens/model_bridge/supported_architectures/cohere.py` | Interleaved RoPE, QK-norm submodules, single-application `logit_scale` |
| `transformer_lens/model_bridge/generalized_components/position_embeddings_attention.py` | `_apply_rotary_pos_emb_interleaved_cos_sin` helper and its dispatch, `per_head` QK-norm phase |
| `transformer_lens/model_bridge/architecture_adapter.py` | `finalize_weight_processing` no-op base hook |
| `transformer_lens/model_bridge/bridge.py` | Calls the hook at the end of `process_weights` |
| `transformer_lens/model_bridge/sources/transformers.py` | Maps HF `use_qk_norm` onto the declared config field |
| `transformer_lens/benchmarks/component_outputs.py` | Per-head QK norms get a `[B, S, H, d_head]` test input |
| `transformer_lens/model_bridge/supported_architectures/AGENTS.md` | Guidance on reimplemented attention and degenerate parity models |
| `tests/unit/model_bridge/supported_architectures/test_cohere_adapter.py` | Rotation, QK-norm, and `logit_scale` unit tests |
| `tests/integration/model_bridge/test_cohere_adapter.py` | Offline parity tests (`d_head = 8`, GQA, with and without QK-norm) |

### Later code changes on the branch

| Commit | Files | Change |
|---|---|---|
| `68dbd36`, `56ddd53` | `tests/integration/model_bridge/test_cohere_adapter.py` | Real-checkpoint acceptance test, then its `<= 1e-4` assertion and intermediate-activation checks |
| `1008f60` | `transformer_lens/config/transformer_bridge_config.py`, `position_embeddings_attention.py`, `cohere.py`, both Cohere test files | RoPE switch moved to the `rotary_interleaved_cos_sin` config field, matching upstream PR #1873 |
| `78278ef` | `transformer_lens/tools/model_registry/generate_report.py`, `architecture_adapter.py` (docstring) | Report descriptions for DeepSeek V2 / V3 and Gemma 3 multimodal; Cohere description narrowed |
| `078c037` | `tests/integration/model_bridge/test_cohere_adapter.py` | HF reference fixture pinned to float32 + eager |
| `a0cc546` | `makefile` | `make cohere-acceptance-test` |

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

- `position_embeddings_attention.py` is used by many architectures. The view/transpose block was restructured and RoPE is now selected through the `rotary_interleaved_cos_sin` config flag. By reading, the default path is equivalent; this is not confirmed by any test.
- A `[n_heads, d_head]` QK-norm weight now selects `per_head` instead of `post_reshape`. Unresolved whether any other adapter, in-tree or out-of-tree, wraps a norm with a 2-D weight; this was not checked exhaustively.
- `TransformerBridge.process_weights` now calls `finalize_weight_processing` for every adapter. The base implementation is a no-op.
- `get_random_inputs` in the attention bridge falls back to an identity rotation (`cos = 1`, `sin = 0`) when the rotary module is unset or raises. Verified in source. Component-level benchmarks on that path cannot detect a RoPE convention error; only full-model parity can.

## 5. Upstream #1126 and #1127

```{admonition} Correction, 2026-10-09
:class: warning
The numbers were probably meant differently from how this section reads them. Notification emails in the project mailbox, each headed `[TransformerLensOrg/TransformerLens]`, describe:

- **#1124** — pull request "Add Command-R adapter": "Adds a Command-R weight-conversion table and acceptance test. Logits match to 1e-4 on the smallest checkpoint."
- **#1126** — issue "bug: ActivationCache repr is very slow for large caches". Unrelated to Command-R or any adapter.
- **#1127** — issue "question: roadmap for Command-R and DeepSeek adapters": "Are Command-R (#1124) and a DeepSeek adapter on the near-term roadmap?"

Verified: those emails exist and say this. Also verified: on github.com, upstream #1126 and #1127 are the merged pull requests analysed below, with different titles and dates. The two sources disagree, and this repository has no issues or pull requests of its own. **Unresolved:** which tracker the emails refer to, and whether the #1124 claim of a 1e-4 match was ever backed by a saved log.

Impact under the email reading: #1126 has no bearing on this release. #1127 is a roadmap question whose honest answer is in section 8: Command-R and DeepSeek V2 / V3 / R1 are implemented but unverified. #1124's claim is not supported by anything in this repository; the registry shows the only non-degenerate Cohere checkpoint failing parity. The analysis below is kept because it is accurate about upstream, but it should not be read as the impact of the issues above.
```

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

## 6. Model-support claims versus implementation

Reconciled on 2026-10-09 by reading the adapters, the three registration sites, `ARCHITECTURE_DESCRIPTIONS`, and `supported_models.json` (generated 2026-06-25; 12,899 entries, 65 architectures, 1,017 verified). `docs/source/_static/supported_models.json`, which feeds the published support matrix, is byte-identical to the registry file. Nothing here was executed. Registry counts are the recorded results of earlier `verify_models` runs, not re-verified.

| Family | Adapter file / factory key | Registry (verified / failed / skipped / untested) | Documentation claim | Finding |
|---|---|---|---|---|
| Command-R, Command-R+, Aya | `cohere.py` / `CohereForCausalLM` | 1 / 1 / 17 / 0 | "Cohere's Command models"; release notes list Cohere as supported | Registered in all sites. Overstated: no real checkpoint is verified, and on `main` logits diverge from HF for `d_head > 2` (section 1). Command R7B and Command A use `Cohere2ForCausalLM`, which has no adapter. Description narrowed accordingly. |
| DeepSeek V2 | `deepseek_v2.py` / `DeepseekV2ForCausalLM` | 0 / 1 / 29 / 0 | Covered by "DeepSeek V2/V3/R1" MLA notes | Registered, but **zero verified models**. The one model that ran (`yujiepan/deepseek-v2-0628-tiny-random`) failed Phase 1 at 0%. Had no report description; added. |
| DeepSeek V3 | `deepseek_v3.py` / `DeepseekV3ForCausalLM` | 2 / 5 / 53 / 0 | Same | Registered. Both verified entries are tiny random models, each "with issues" (`P3=89.5%`). Five tiny models failed Phase 1, including both `trl-internal-testing` ones. No full-size checkpoint has run. Had no report description; added. |
| DeepSeek R1 | No separate adapter; R1 loads as `DeepseekV3ForCausalLM` | 0 verified of 10 R1-named V3 entries (all skipped: size or quantization) | "DeepSeek V2/V3/R1" | The routing claim is accurate: the adapter docstring and the registry both put R1 on the V3 architecture. R1 itself has never been verified. Most R1-named registry entries are distillations on other architectures (74 Qwen2, 34 Llama) and say nothing about this adapter. |
| Gemma 1 / 2 / 3 | `gemma1.py`, `gemma2.py`, `gemma3.py` | 26 / 2 / 14 / 36; 24 / 0 / 51 / 32; 21 / 19 / 22 / 120 | Several docs said the `√d_model` embedding scale is folded into `embed.weight` via `preprocess_weights` | **Documentation was wrong.** No Gemma adapter defines `preprocess_weights`; each says the scale is applied at runtime by the wrapped HF embedding module and must not be pre-scaled. Docs corrected. Gemma 3 has 19 failed entries. |
| Gemma 3 multimodal, 3n, 4 | `gemma3_multimodal.py`, `gemma3n.py`, `gemma4.py` | 11 / 6 / 17 / 21; 1 / 3 / 0 / 0; 7 / 0 / 16 / 33 | Described as text-only for 3n and 4 | `Gemma3ForConditionalGeneration` had no report description; added. `Gemma4UnifiedForConditionalGeneration` routes to `gemma4.py` and has zero verified models (18 skipped). |
| T5Gemma | `t5gemma.py` | 0 / 20 / 0 / 6 | Listed as supported | Registered, but every model that ran failed. |

Other verified discrepancies:

- `ARCHITECTURE_DESCRIPTIONS` lacked entries for 24 of the 66 architectures in `HF_SUPPORTED_ARCHITECTURES`. The docs said a missing entry drops the architecture from the generated table; the code falls back to an auto-generated description instead. Docs corrected. Three entries were added (above); the other 21 are still missing.
- `tests/unit/model_bridge/supported_architectures/` has 57 adapter test files, not 26, and none for `deepseek_v2.py` or `deepseek_v3.py`, although the adapter guide calls a unit adapter test mandatory. Count corrected; the tests were not written.
- `debugging_numerical_divergence.md` pointed at `cfg.scale_embeddings`, which does not exist anywhere under `transformer_lens/`. Corrected.
- `NemotronHForCausalLM` has an adapter and is in `HF_SUPPORTED_ARCHITECTURES`, yet has no registry entries and is still listed in `architecture_gaps.json` as unsupported. Both data files are generated, so they were left alone; they need a fresh scrape.

Left unchanged on purpose: `news/release-3.0.md` (a dated release essay), the registry and gaps data files (only `verify_models` and the scraper may write them), and the "DeepSeek V2 / V3 / R1" MLA notes in `compatibility_mode.md` and `migrating_to_v3.md`, which describe block structure and match the code.

## 7. Command-R release-verification matrix

First compiled at `78278ef` on 2026-10-09. No status below has changed since, because no test has run; later commits added the real-checkpoint acceptance test (A4), pinned the HF reference to float32 for A6, and added `make cohere-acceptance-test`, which runs A4–A7 in one command. Ordered by release priority: real-model logit parity first, then registry consistency, then compatibility with existing adapters.

**No row has a runtime PASS.** Evidence is tagged by kind, and the kinds must not be mixed:

- **[V] Verified now** — command output or source read directly during this work. Static only; nothing was executed against `torch`.
- **[H] Historical** — recorded in `transformer_lens/tools/model_registry/data/supported_models.json` by earlier `verify_models` runs, all predating the fixes in `1b7c4a9`. Not re-verified.
- **[U] Unexecuted** — the check exists or is specified but has never produced a result on this branch.

Status values: **PASS (static)** means confirmed by reading only. **FAIL** means evidence shows the requirement is not met. **BLOCKED** means it cannot run without something missing. **NOT RUN** means runnable in a normal dev environment but not yet executed.

### A. Real-model logit parity (highest priority)

| ID | Required check | Files / command | Evidence | Status | Release impact | Next action |
|---|---|---|---|---|---|---|
| A1 | Phase 1 logit parity on a real Command-R checkpoint (`CohereLabs/c4ai-command-r-v01` or `-08-2024`) | `transformer_lens/tools/model_registry/verify_models.py` → `--model CohereLabs/c4ai-command-r-v01` | [H] Both entries skipped 2026-06-25: "Config unavailable: Gated repo". No Command-R checkpoint has ever produced a parity score. | **BLOCKED** (gated repo needs `HF_TOKEN` with licence accepted; 35B-class model needs memory above the 70 GB default limit) | **Release blocker.** Without it the "Command-R supported" claim has no evidence. | Accept the licence, source `.env`, run with `--dry-run` first, then for real with a raised `--max-memory`. |
| A2 | Phase 1 parity on the smallest real Cohere-architecture checkpoint (`CohereLabs/aya-23-8B` or `aya-expanse-8b`, 8.0B params) | Same tool, `--model CohereLabs/aya-23-8B` | [H] Skipped as gated. The ungated mirror `adamo1139/aya-expanse-8b-ungated` was skipped: "Estimated 79.0 GB exceeds 70.0 GB limit". | **BLOCKED** (token, memory) | Blocker if A1 cannot run; otherwise strong supporting evidence. Cheapest real weights with `d_head > 2`. | Run on a machine with ≥ 80 GB, or with a reduced `--dtype`. |
| A3 | Phase 1 parity on a real QK-norm checkpoint (Command-R+: `CohereLabs/c4ai-command-r-plus`) | Same tool | [H] Skipped as gated. [V] The QK-norm code path exists only on this branch. Unresolved: the config is gated, so `use_qk_norm=True` for this checkpoint was not confirmed here. | **BLOCKED** (token; 104B params) | Blocker for advertising Command-R+ specifically. Not a blocker for Command-R alone. | Confirm `use_qk_norm` from the config, then run A1-style. If infeasible, state Command-R+ as unverified in the release notes. |
| A4 | Phase 1 parity on a non-degenerate ungated tiny model (`hyper-accel/tiny-random-cohere`; config on the Hub shows hidden 512, 4 heads, so `d_head = 128`) | Same tool, `--model hyper-accel/tiny-random-cohere` (add `--reverify` if it is skipped as already tested); and `tests/integration/model_bridge/test_cohere_adapter.py::TestCohereRealCheckpointLogitParity` (`-m slow`) | [H] **FAILED** 2026-06-26, before the fix: `P1=50.0% (failed: forward_pass_logits)`, `max_diff=0.009886, mean_rel=0.980186`. | **FAIL** (historical, pre-fix). Re-run on this branch: **NOT RUN**. | **Release blocker.** This is the fastest real check that the RoPE fix works; it needs no token and little memory. | Run this first. Must reach P1 = 100%. |
| A5 | Offline parity on synthetic non-degenerate models (`d_head = 8`, GQA, with and without QK-norm): logits, attention patterns, processed logits | `tests/integration/model_bridge/test_cohere_adapter.py` — `TestCohereOfflineParity`, `TestCohereOfflineProcessedLogitScale` | [U] Written on this branch, never executed. `pytest` stopped at `import torch`. | **BLOCKED** here (no `torch`); NOT RUN elsewhere | Blocker. Necessary but **not sufficient**: random weights do not substitute for A1–A4. | `uv run pytest tests/integration/model_bridge/test_cohere_adapter.py -v -k Offline` |
| A6 | Parity on the Hub tiny model (`trl-internal-testing/tiny-CohereForCausalLM`), including the new processed-logits test | Same file — `TestCohereForwardEquivalence`, `TestCohereLogitScaleEndToEnd`, `TestCohereTiedEmbedding` | [H] Verified with issues 2026-04-15: P1 100, P2 100, P3 94.7 (`weight_modification` failed), P4 72.3. [V] This model has `d_head = 2`, where both RoPE conventions coincide, so its P1 = 100 cannot detect the defect. | **NOT RUN** on this branch | Regression guard only. A pass here proves little about real models. | Run with the rest of the file; investigate if P3 `weight_modification` still fails after the `logit_scale` fix. |
| A7 | Rotation, QK-norm declaration, and single-application `logit_scale` unit tests | `tests/unit/model_bridge/supported_architectures/test_cohere_adapter.py` — `TestCohereRotaryConvention`, `TestCohereQKNorm`, `TestCohereLogitScaleRuntimeNeutralized` | [U] Never executed. [V] A plain-Python re-derivation showed the conventions differ by 0 at `d_head = 2` and by 2.62 / 1.38 at 4 / 8; that checks the arithmetic, not the code. | **BLOCKED** here; NOT RUN elsewhere | Blocker (part of `make unit-test`). | `uv run pytest tests/unit/model_bridge/supported_architectures/test_cohere_adapter.py -v` |

### B. Registry consistency

| ID | Required check | Files / command | Evidence | Status | Release impact | Next action |
|---|---|---|---|---|---|---|
| B1 | `CohereForCausalLM` registered in all four sites | `transformer_lens/model_bridge/supported_architectures/__init__.py`, `transformer_lens/factories/architecture_adapter_factory.py`, `transformer_lens/tools/model_registry/__init__.py`, `transformer_lens/tools/model_registry/generate_report.py` | [V] Present in each file by reading. [U] `tests/unit/tools/test_model_registry.py::TestRegistrySyncedWithFactory` not run. | **PASS (static)** | Low risk. | Run the sync test with the unit tier. |
| B2 | Registry statuses reflect the code being released | `transformer_lens/tools/model_registry/data/supported_models.json` | [V] All 19 Cohere entries are dated 2026-04-15 to 2026-06-26, before `1b7c4a9`. The one "verified" entry was produced by code with the defects in section 1. | **FAIL** (stale) | **Release blocker.** The published matrix would show a verification that the current code has not earned. | `verify_models --reverify --architectures CohereForCausalLM`. Never hand-edit. |
| B3 | Real checkpoints covered by verification | Same file | [H] 17 of 19 skipped: 10 gated, 4 over the memory limit, 3 quantized formats. 1 verified (degenerate tiny), 1 failed. | **FAIL** | Blocker for an unqualified support claim. | Resolve through A1–A4. |
| B4 | Published support matrix matches the registry | `docs/source/_static/supported_models.json` | [V] Byte-identical to the registry file (`cmp`). | **PASS (static)** | None today; it will go stale the moment B2 is fixed. | Regenerate the docs copy after re-verification. |
| B5 | Scope of the claim matches the adapter | `transformer_lens/tools/model_registry/generate_report.py`, `transformer_lens/tools/model_registry/data/architecture_gaps.json` | [V] `Cohere2ForCausalLM` (Command R7B, Command A; 27 models) is listed as an unsupported gap and has no adapter. Description narrowed in `78278ef`. `news/release-3.0.md` still lists "Cohere" without qualification. | **PASS (static)** for the report; open for the release notes | Medium: users may assume newer Command models work. | State the Cohere2 exclusion in the next release notes. |
| B6 | Cohere config values reach the bridge config | `transformer_lens/model_bridge/sources/transformers.py`, `transformer_lens/model_bridge/sources/_bridge_builder.py`, `transformer_lens/config/transformer_bridge_config.py` | [V] `logit_scale` and `rope_parameters` are in both passthrough lists; `use_qk_norm` is a declared field with an explicit handler. [U] The runtime assertions (`test_cfg_logit_scale_matches_hf`, `test_cfg_rope_parameters_matches_hf`, `test_fixture_exercises_rope_and_gqa`) were not run. | **PASS (static)**; runtime NOT RUN | Blocker if it fails: a dropped `use_qk_norm` makes Command-R+ fail at boot. | Covered by A5 and A6. |

### C. Compatibility with existing adapters

| ID | Required check | Files / command | Evidence | Status | Release impact | Next action |
|---|---|---|---|---|---|---|
| C1 | Default attention path unchanged for the other adapters on the shared bridge | `transformer_lens/model_bridge/generalized_components/position_embeddings_attention.py`; 28 adapter files reference it (27 besides `cohere.py`) | [V] `rotary_interleaved_cos_sin` defaults to `False` and that branch calls the same Llama function as before; the view/transpose restructuring reads as equivalent. A second static review agreed. [U] No adapter test has run. | **NOT RUN** | **Release blocker.** This is the widest blast radius on the branch: Llama, Mistral-family, Gemma 1–3, Qwen 2–3, Phi, OLMo and others. | `make test-pr`. |
| C2 | QK-norm phase selection unchanged for adapters that already declare `q_norm` on this bridge | Same file (`_decide_qk_norm_phase`); `apertus.py`, `gemma3.py`, `gemma3_multimodal.py`, `glm4_moe.py`, `hunyuan_v1_dense.py`, `olmo2.py`, `olmoe.py`, `qwen3.py`, `qwen3_moe.py` under `transformer_lens/model_bridge/supported_architectures/` | [V] Only a 2-D `[n_heads, d_head]` weight now selects the new `per_head` phase; 1-D weights take the same branches as before. Unresolved: the actual weight shapes for these nine adapters were not inspected at runtime. | **NOT RUN** | Blocker if any of the nine has a 2-D norm weight; otherwise none. | Run their unit and integration tests; they are in `make test-pr`. |
| C3 | New `finalize_weight_processing` hook is inert for every other adapter | `transformer_lens/model_bridge/architecture_adapter.py`, `transformer_lens/model_bridge/bridge.py` (end of `process_weights`) | [V] Base method is a no-op; `cohere.py` is the only override (grep). The call is guarded with `hasattr`. | **PASS (static)**; runtime NOT RUN | Low, but `process_weights` runs for every model in compatibility mode. | Covered by `make test-pr` (14 test files reference `process_weights`). |
| C4 | Component benchmark change does not alter other architectures | `transformer_lens/benchmarks/component_outputs.py` | [V] The new branch triggers only when a `q_norm` / `k_norm` weight is 2-D. | **PASS (static)**; runtime NOT RUN | Low. Affects `verify_models` Phase 1 component scores, not library behaviour. | Observed as part of A4 and D2. |
| C5 | Format and type gates with the pinned tools | `make check-format`, `uv run mypy .` | [V] Unpinned black / ruff / mypy found nothing new versus `main` (section 2). Indicative only: wrong versions, `torch` absent. | **NOT RUN** | CI blocker (`format-check`, `type-check`). | Run both; fix anything reported. |
| C6 | No numerical regression in representative verified models that share the bridge | `verify_models --reverify` on one verified model each for Llama, Gemma 3, Qwen 3 and OLMo 2 | [H] Existing verified entries for these families predate the branch. | **NOT RUN** | Recommended before release; blocker if C1 or C2 shows any failure. | Run sequentially, never in parallel. |

### D. Reading the matrix

1. **Counts.** 19 checks: 0 runtime PASS, 6 PASS (static only: B1, B4, B5, B6, C3, C4), 3 FAIL (A4 historical, B2, B3), 3 BLOCKED on external access (A1–A3), 7 not run (A5, A6, A7, C1, C2, C5, C6).
2. **Fastest path to a first real signal:** A7 → A5 → A4. None needs a token or a large machine, and A4 is a real Hub model that failed before the fix.
3. **Minimum to lift the NO-GO for Command-R:** A4, A5, A7, C1, C2, C5 passing, B2 regenerated, plus A1 or A2. Command-R+ additionally needs A3 or an explicit "unverified" label.

### E. Acceptance test audit

**Requirement source.** [V] The repository contains no Command-R-specific acceptance document (the project's written methodology was found later, outside the repository; see section 8). The governing requirement is the general new-adapter rule: an integration parity test at `tests/integration/model_bridge/test_<arch>_adapter.py` asserting logit parity against Hugging Face at fp32 with eager attention (`AGENTS.md` §11, `supported_architectures/AGENTS.md` "Required tests"), with `< 1e-4` max absolute difference as the fp32 bar (`debugging_numerical_divergence.md`). `tests/acceptance/` holds no Cohere test and, by convention, is not where per-adapter parity lives.

| Test | Compares bridge vs HF logits at 1e-4 max abs? | Genuine? |
|---|---|---|
| `TestCohereForwardEquivalence::test_forward_matches_hf` (on `main`) | Yes, formally: `max_diff < 1e-4` against an independently loaded HF model. | **No.** [V] The model has `d_head = 2`, so a wrong RoPE convention cannot change its logits. On `main` the HF reference fixture was unpinned and loaded in float16 (the checkpoint's declared dtype) against a float32 bridge; `078c037` pins it to float32 + eager. There is no check that the logits are larger than the tolerance; upstream measured `2.2e-09` on this model with the defect present. |
| `TestCohereOfflineParity::test_forward_matches_hf` (added in `1b7c4a9`) | Yes: `max_diff < 1e-4`, with a guard that HF logits exceed `1e-2`. | Partly. Sensitive to RoPE, GQA and QK-norm, but on random synthetic weights, not a checkpoint. |
| `TestCohereRealCheckpointLogitParity::test_logits_match_hf_within_tolerance` (added with this audit) | Yes: `max_diff < 1e-4` at fp32 + eager on `hyper-accel/tiny-random-cohere`, at two sequence lengths, with a guard that HF logits exceed `100 ×` the tolerance. | Intended as the acceptance gate. Real ungated checkpoint with `d_head = 128`. Marked `slow` because the model is outside the CI cache (per `tests/AGENTS.md`), so `make integration-test` does **not** run it; it must be invoked with `-m slow`. |

**Execution status: BLOCKED, not run.** [V] `pytest tests/integration/model_bridge/test_cohere_adapter.py -m slow -k RealCheckpoint` exited 4 at `tests/conftest.py:15: ModuleNotFoundError: No module named 'torch'`. `pypi.org` returned 403 and `huggingface.co` was unreachable from the sandbox, so neither `torch` nor the checkpoint can be obtained there. Whether the new test passes is unknown.

**Upstream has already fixed the RoPE defect.** [V, from the PR page only; its diff was not read] `TransformerLensOrg/TransformerLens` PR #1873, "fix(bridge): rotate Cohere and Cohere2 with their own RoPE convention", was merged into `dev` on 2026-10-09. Its author reports, for `hyper-accel/tiny-random-cohere` at fp32 on CPU, a bridge-versus-HF max difference of `1.07e-02` before and `7.8e-08` after. That is independent runtime confirmation of defect 1 and of this checkpoint as a discriminating gate. It is evidence about upstream code, not about this branch. Consequences:

- **Resolved after comparing against the PR diff (below).** This branch originally duplicated that fix with a different design (a `rotary_interleaved` constructor argument). It now uses upstream's.
- Upstream also has a Cohere2 adapter and tests, which this repository lacks, so this repository is behind upstream `dev`.
- The PR author did not run `verify_models` or any gated checkpoint, so rows A1–A3 remain open upstream as well. Whether upstream addresses QK-norm or the `logit_scale` double application (defects 2 and 3) is unresolved.

### F. Reconciliation with upstream PR #1873

[V] Based on reading the PR diff (5 files: `transformer_bridge_config.py`, `position_embeddings_attention.py`, `cohere.py`, and the Cohere and Cohere2 unit tests). Nothing was executed; the tests below are **not run**.

| Area | Upstream #1873 | This branch before | Conflict | Action taken |
|---|---|---|---|---|
| RoPE rotation | Config field `rotary_interleaved_cos_sin` (default `False`), module-level `_apply_rotary_pos_emb_interleaved_cos_sin`, dispatch via `getattr(self.config, ...)`; `cohere.py` sets `self.cfg.rotary_interleaved_cos_sin = True` | Constructor argument `rotary_interleaved` and a `_apply_rotary` method on the bridge | **Duplicate, conflicting.** Same arithmetic, different names and mechanism, in the same lines of two files. Syncing with upstream would have left two switches for one behaviour. | Replaced with upstream's field name, helper name and body, and adapter line. This also matches the repo convention of driving the attention bridge from config flags (`gated_q_proj`, `use_split_qkv_input`). When the flag is `False` the same Llama function is imported and called as on `main`. |
| QK-norm (`use_qk_norm`) | Not touched. Upstream's parity tests use the default `use_qk_norm=False`. | Adapter declares `q_norm` / `k_norm`; new `per_head` phase | None. Additive. | Kept. Still unverified by either side. |
| `logit_scale` after `process_weights` | Not touched. Upstream's parity tests never call `process_weights`. | `finalize_weight_processing` hook resets the HF runtime multiply | None. Additive. | Kept. Still unverified by either side. |
| Tests | Unit-tier `TestCohereAdapterConfig::test_rotary_interleaved_cos_sin`, `..._defaults_to_false`, and `TestCohereParity::test_logits_match_hf` (synthetic `d_head = 8` model, widened weights, batch of 2) using an `assert_tiny_parity` helper | Integration-tier offline parity only | The upstream helper `assert_tiny_parity` does not exist in this repository. | Added the three tests under the same names in the unit file, with an explicit `< 1e-4` assertion in place of the missing helper. Existing rotation unit tests now call the module-level helper. |
| Cohere2 | Parity test for `Cohere2ForCausalLM`; its adapter inherits the flag | No Cohere2 adapter exists here | Not applicable. | None. Out of scope; this repository is behind upstream `dev`. |

Differences that remain against upstream, all unresolved until the two are actually merged:

- This repository's `position_embeddings_attention.py` has no `rotary_adjacent_pairs` branch or `_apply_rotary_pos_emb_adjacent_pairs` helper, which upstream's dispatch chains after the new flag. A merge will still need a manual resolution there.
- The partial-rotary path here applies the selected rotation to the rotated slice. How upstream handles that case is not visible in the diff.
- One fixture detail changed for safety: the offline integration fixture no longer passes `attn_implementation="eager"` to `CohereConfig(...)` and relies on setting `_attn_implementation` after construction, the pattern `test_glm_moe_dsa_adapter.py` already uses.

## 8. Audit of remaining criteria (2026-10-09, branch at `1008f60` plus this commit)

### Acceptance methodology found outside the repository

[V] A project note in Drive, "Weight-conversion acceptance test methodology", states the rule this release is held to. It is not in the repository. Compared with the test on this branch:

| Methodology requirement | `TestCohereRealCheckpointLogitParity` | Status |
|---|---|---|
| Smallest **public** checkpoint of the family | `hyper-accel/tiny-random-cohere` (ungated). `trl-internal-testing/tiny-CohereForCausalLM` is smaller but has `d_head = 2` and cannot detect a RoPE error. | Met, with a documented substitution |
| Load in HF and in **HookedTransformer** | Loads HF and **TransformerBridge**. [V] No Cohere entry exists in `transformer_lens/supported_models.py` or `loading_from_pretrained.py`, so a HookedTransformer comparison is impossible in this codebase. | **Deviation**; needs a maintainer decision |
| Fixed short prompt, float32 | Fixed token ids (BOS plus fixed in-vocabulary ids), float32, eager attention | Met; uses ids instead of text to avoid a tokenizer dependency |
| Max absolute logit delta `<= 1e-4` | `max_diff <= 1e-4`, with a guard that HF logits exceed `1e-2` | Met as written |
| Secondary: `resid_pre` layer 0, attention pattern layer 0, final resid | `test_intermediate_activations_match_hf`, added in this commit | Met as written |
| Save the log; record pass/fail and the box in the adapter matrix and `Releases/demo_sweep_results.csv` | No log exists because the test has not run. `Releases/` does not exist in this repository. | **Not met** |
| No adapter merges without the test | The Cohere adapter is already on `main` without a passing record | **Not met** (pre-existing) |

**Execution: BLOCKED, not run.** `pytest ... -m slow` again stopped at `ModuleNotFoundError: No module named 'torch'`.

### Actual CI result for `main`

[V] GitHub Actions run 37984279901, workflow "Checks", commit `a4c3648` (`main`), finished 2026-10-09 20:09 UTC, conclusion **failure**. This is the baseline *before* any change on this branch.

| Job | Result |
|---|---|
| Format Check | passed |
| Type Check | **failed** (step "Type check") |
| Compatibility Checks 3.10 / 3.11 / 3.12 | **failed** (step "Unit Test") |
| Docstring Test | **failed** |
| Full Code Coverage Test | **failed** |
| Benchmark Tests | **failed** |
| MPS Checks | **failed** (step "MPS Unit Tests") |
| Notebook Checks | 2 passed (`LLaMA`, `LLaMA_GPU_Quantized`), 13 **failed** |
| Build Docs, Deploy Docs | skipped |

Unresolved: **why** they failed. The job logs could not be downloaded from the sandbox, and the annotations show only exit codes and a `git` exit-128 warning on every job. No CI run exists for this branch: the workflow triggers on pushes to `main` / `dev*` / `refactor*` and on pull requests, and no pull request has been opened. Opening one is the quickest way to get real test results for this branch, but with `main` already red its result will need comparing against this baseline.

### Support status by family

Three states, used consistently here, in `Adapters/adapter_support_matrix.xlsx`, and in `model_tables.md`:

- **Implemented and verified** — adapter exists and a passing parity result is recorded for a non-degenerate checkpoint.
- **Implemented, unverified** — adapter exists; no such result is recorded.
- **Unsupported** — no adapter.

| Family | State | Basis |
|---|---|---|
| Command-R, Command-R+, Aya (`CohereForCausalLM`) | Implemented, unverified | Section 7. One verified registry entry, on a degenerate `d_head = 2` model; the non-degenerate one failed. |
| DeepSeek V2 (`DeepseekV2ForCausalLM`) | Implemented, unverified | 0 verified, 1 failed, 29 skipped. No unit adapter test. |
| DeepSeek V3 (`DeepseekV3ForCausalLM`) | Implemented, unverified | 2 tiny random models verified with issues; 5 tiny models failed Phase 1; no full-size run. No unit adapter test. |
| DeepSeek R1 | Implemented, unverified | Loads through the V3 adapter. 0 of 10 R1-named V3 entries verified. |
| Command R7B, Command A (`Cohere2ForCausalLM`, `Cohere2MoeForCausalLM`) | Unsupported | Listed in `architecture_gaps.json`; no adapter here. |
| DeepSeek V3.2, V4, original MoE, VL2 (`DeepseekV32…`, `DeepseekV4…`, `DeepseekForCausalLM`, `DeepseekVLV2…`) | Unsupported | Listed in `architecture_gaps.json`; no adapter here. |

### Support matrix file

[V] `Adapters/adapter_support_matrix.xlsx` did not exist in this repository or its history. A Google Sheet titled "Model-adapater support matrix.xlsx" exists in the project Drive with 13 families, all marked `supported`, and no Command-R or DeepSeek rows. The repository file was created in this commit from that sheet: its 13 rows and its `coverage` tab are copied unchanged, and six rows were added (Command-R; DeepSeek V2, V3, R1; and two unsupported groups) with a status key. The Drive sheet itself was not modified. The copied rows' acceptance results were **not** verified. A later pass added a "repository check" column recording what this repository shows for every row. For the 13 copied rows: 8 checkpoints are in the HookedTransformer list (`transformer_lens/supported_models.py`), of which 7 are also verified in the Bridge registry (`gpt2` under the id `openai-community/gpt2`) and one (`EleutherAI/gpt-neox-20b`) is skipped there; 2 rows are **contradicted** (LLaMA-4 and Qwen-2.5-VL: checkpoint in neither list, architecture listed as unsupported in `architecture_gaps.json`); 2 have a checkpoint id that matches nothing exactly (Gemma-3, Phi-3; the `-it` / `-pt` / `-instruct` variants exist and are verified); 1 is unconfirmed (Mistral-Small-3). The "since TL version" values cannot be checked: this repository's package version is `0.0.0`. The header line "Every family ships a weight-conversion acceptance test" is therefore unconfirmed, and false as written for at least the two contradicted rows. The `coverage` counts were left as they were; no family was added to the supported total.

### Other criteria

| Criterion | Evidence | Status |
|---|---|---|
| Changes committed and pushed | `git ls-remote` matches local `HEAD` after each commit | Met |
| Runtime parity and regression tests pass | None executed | **Not met** |
| Registry re-verified for Cohere | Entries predate the fix | **Not met** |
| Sync with upstream `dev` (PR #1873, Cohere2) | Section 7F | **Not met** |
| Maintainer notified | An unsent Gmail draft exists, addressed to `bram.giraffetasks@gmail.com` at the repository owner's choice. No source lists that address as a maintainer, and no maintainer email is listed in this repository, in upstream `pyproject.toml`, or in the project mailbox or Drive. Named maintainers: Jonah Larson (primary, per `news/release-3.0.md`), Bryce Meyer (previous). | **Not met**: nothing sent; recipient unverified |

(fastest-path)=
## 9. Fastest path to release readiness

### Blocker, re-checked 2026-10-09 18:25 (UTC-5)

The sandbox this work was done in **cannot run any Command-R test**, unit or offline. Checked directly:

| Probe | Result |
|---|---|
| `import torch`, `import transformers` under Python 3.11, 3.12 and 3.13 | `ModuleNotFoundError` on all three |
| Search of the filesystem and both `uv` caches for a `torch` or `transformers` install or wheel | None found |
| `pip download torch` | "No matching distribution found" |
| `curl https://pypi.org/simple/torch/` | HTTP 403: "Host not in allowlist: pypi.org. Add this host to your network egress settings to allow access." |
| `files.pythonhosted.org`, `download.pytorch.org`, `huggingface.co`, `conda.anaconda.org` | Refused or unreachable |
| Docker | No daemon running |
| Linked workstation | Not connected |

No code was changed in response. Nothing below has been run.

**To unblock, any one of:** allow `pypi.org` and `files.pythonhosted.org` for the session (add `huggingface.co` for steps 4, 5, 8 and 9); run the steps on a developer machine; or open a pull request from the branch so GitHub Actions runs steps 2, 3, 5, 6 and 7 (it will not run step 4, which is marked `slow`; use `make cohere-acceptance-test` for that).

### Ordered checks

**Shortcut for steps 2–5:** `make cohere-acceptance-test` runs both Cohere test files in full — unit, offline parity, Hub tiny model and the real-checkpoint acceptance test — including `slow` tests and with no automatic reruns. It needs `torch` and Hugging Face access. Pass criterion: 0 failed, 0 errors; keep the output as the acceptance log. [V] `make -n cohere-acceptance-test` resolves to the intended `pytest` command; running it here fails at dependency install (403 from `files.pythonhosted.org`).

Run top to bottom and stop at the first failure. Steps 1–3 need no Hugging Face access and take minutes; they are the fastest real signal. Owners are roles: no one has been assigned or has accepted any of these, and the names in brackets are proposals.

| # | Check | Command | Owner | Pass criterion |
|---|---|---|---|---|
| 0 | Understand why CI on `main` is red | Read the logs of Actions run 37984279901 | Repository owner [Neil] | Cause of each failing job identified; either fixed or recorded as a known baseline failure with a reason |
| 1 | Environment builds | `uv sync` | Branch owner [Neil] | Exit code 0 |
| 2 | Cohere unit tests, including offline parity | `uv run pytest tests/unit/model_bridge/supported_architectures/test_cohere_adapter.py -v` | Branch owner | 0 failed, 0 errors. `TestCohereParity::test_logits_match_hf` reports max abs difference `< 1e-4` |
| 3 | Offline integration parity (GQA, QK-norm, processed logits) | `uv run pytest tests/integration/model_bridge/test_cohere_adapter.py -v -k Offline` | Branch owner | 0 failed. Both `no_qk_norm` and `qk_norm` variants pass; `test_llama_rotation_diverges_from_hf` passes (the fixture can still detect the bug) |
| 4 | **Acceptance test** on a real public checkpoint | `uv run pytest tests/integration/model_bridge/test_cohere_adapter.py -v -m slow` | Branch owner | 0 failed. Max abs logit delta `<= 1e-4` at both sequence lengths; layer-0 `resid_pre`, layer-0 attention pattern and final resid each `<= 1e-4`. Log saved |
| 5 | Remaining Cohere integration tests (Hub tiny model) | `uv run pytest tests/integration/model_bridge/test_cohere_adapter.py -v` | Branch owner | 0 failed |
| 6 | No regression in adapters sharing the attention bridge | `make test-pr` | Branch owner | 0 failed. Any failure must be reproduced on `main` at `a4c3648` and explained before it is set aside (`AGENTS.md` §10) |
| 7 | Format and types, pinned tools | `make check-format` and `uv run mypy .` | Branch owner | Both exit 0 |
| 8 | Registry verification, ungated model | `uv run python -m transformer_lens.tools.model_registry.verify_models --model hyper-accel/tiny-random-cohere` | Branch owner | Phase 1 = 100%, status verified; registry file updated by the tool |
| 9 | Registry verification, real checkpoint | Same tool, `--model CohereLabs/aya-23-8B` (or a Command-R checkpoint), after `--dry-run` | Someone with a Hugging Face token that has accepted the Cohere licence and about 80 GB of memory. **Unassigned** | Phase 1 = 100%, status verified |
| 10 | Decide: acceptance against TransformerBridge instead of HookedTransformer | Section 8 | A maintainer. **Unassigned**; raised in the unsent draft | Written decision |
| 11 | Decide: take upstream PR #1873 by syncing with `dev`, or keep this branch's equivalent | Section 7F | A maintainer. **Unassigned** | Written decision; no duplicate RoPE switch after the sync |
| 12 | Record the result | `Adapters/adapter_support_matrix.xlsx`; docs copy of the registry | Branch owner | Command-R row changed from "implemented, unverified" to "supported" only after steps 4, 6 and 9 pass, with the log location noted |

**Release gate for Command-R:** steps 1–9 pass and steps 10–12 are closed. Command-R+ additionally needs step 9 repeated on a `use_qk_norm=True` checkpoint, or an explicit "unverified" label.

**If a step fails:** fix on the branch, re-run that step and every earlier one, and do not loosen a tolerance or add a skip to get past it.

## 10. Final audit checklist (2026-10-09)

No written list of 18 release-readiness criteria was found in this repository, the project Drive or the project mailbox. The 18 items below are a reconstruction from the requests made during this work and the written acceptance methodology; replace them if an authoritative list exists. PASS means the evidence was checked directly. No PASS here rests on a test run, because none has run.

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Highest-impact Command-R gap identified and fixed in code | BLOCKED | Fixes for RoPE, QK-norm and `logit_scale` are on the branch; none executed |
| 2 | Changes committed and pushed to the remote branch | PASS | `git ls-remote` equals local `HEAD`; working tree clean |
| 3 | Command-R unit and offline parity tests pass | BLOCKED | `ModuleNotFoundError: No module named 'torch'`; `pypi.org` returns 403 |
| 4 | Format and type checks pass with pinned tools | BLOCKED | `make check-format`, `uv run mypy .` not runnable; unpinned tools show nothing new versus `main` |
| 5 | Release-readiness document records fixes, files, checks, blocked tests and risks | PASS | This file, sections 1–4 |
| 6 | Impact of #1126 / #1127 stated, facts separated from open questions | PASS | Section 5, including the correction; which tracker the numbers belong to is unresolved |
| 7 | Explicit BLOCKED / NO-GO recommendation | PASS | Header and section 11 |
| 8 | Model-support documentation matches the implementation (Command-R, DeepSeek, Gemma) | PASS | Section 6; Gemma fold claims corrected in five documents |
| 9 | Prioritised verification matrix with evidence, status, impact, next action | PASS | Section 7 |
| 10 | Acceptance test exists and compares against Hugging Face at 1e-4 max absolute difference | PASS | `TestCohereRealCheckpointLogitParity`; deviation: TransformerBridge, not HookedTransformer |
| 11 | Acceptance test passes on the smallest public checkpoint, log saved | BLOCKED | Never run; needs `torch` and Hub access |
| 12 | No duplicate or conflicting fix relative to upstream PR #1873 | PASS | Section 7F; flag and helper names match upstream's diff |
| 13 | Command-R row present in `Adapters/adapter_support_matrix.xlsx` | PASS | Row 18, status "implemented, unverified" |
| 14 | Documentation separates implemented, unverified and unsupported families, including DeepSeek V2 / V3 / R1 | PASS | Section 8, `model_tables.md`, matrix rows 19–23 |
| 15 | Command-R registry entries are consistent with the released code and verified | FAIL | 19 entries: 1 verified (a `d_head = 2` model), 1 failed, 17 skipped; all predate the fix |
| 16 | Existing adapter tests pass (no regression) | BLOCKED | Not run on the branch. Baseline is already FAIL: CI run 37984279901 on `main` failed Unit Test on 3.10 / 3.11 / 3.12 and Type Check |
| 17 | Status email exists as a Gmail draft and is unsent | PASS | Draft `r1685509882029817192` carries only the DRAFT label; nothing with that subject in Sent |
| 18 | Draft is addressed to a verified TransformerLens maintainer | FAIL | Addressed to `bram.giraffetasks@gmail.com`, chosen by the repository owner. No source lists this address as a maintainer; the named maintainers have no listed email |

Totals: 11 PASS, 2 FAIL, 5 BLOCKED. (An earlier revision of this line said 10 / 2 / 6, which did not match the table.)

## 11. Recommendation

**BLOCKED / NO-GO** for releasing Command-R support and for merging the shared attention bridge change into a release branch.

### Confirmed

Each item was checked directly: by reading source, by command output, or from a recorded result. None is a test result for this branch.

- The three defects in section 1 are present in the code on `main`, by comparison with the pinned `transformers` 5.8.1 source.
- Upstream confirmed the RoPE defect at runtime and fixed it (PR #1873); this branch uses the same flag and helper. That evidence is about upstream's code.
- The branch is pushed and matches its remote; it changes 19 files relative to `main`.
- The changed Python files parse, and unpinned lint, format and type tools report nothing that is not already on `main`.
- The registry records one verified Cohere entry (a `d_head = 2` model), one failed, and 17 skipped; all predate the fix.
- CI on `main` failed on 2026-10-09 (Type Check, Unit Test on three Python versions, and others) for reasons not yet known.
- The acceptance test exists and matches the written methodology except that it compares against TransformerBridge.
- The support matrix, `model_tables.md` and the report descriptions agree with the adapter factory and `architecture_gaps.json`.

### Not executed

No test of any kind has run on this branch. The environment it was written in has no `torch` and cannot reach PyPI or the Hugging Face Hub.

- Cohere unit tests and offline parity tests.
- The real-checkpoint acceptance test (`make cohere-acceptance-test`).
- Tests for the other adapters that share the attention bridge (`make test-pr`).
- Format and type checks with the pinned tool versions.
- `verify_models` on any Cohere checkpoint, gated or not.
- Anything exercising QK-norm or the `logit_scale` fix at runtime, here or upstream.

### Why this is a NO-GO

The fixes are plausible and one of them is corroborated upstream, but nothing shows that this branch produces correct logits or leaves other adapters unchanged. The decision should be revisited only when the items under "Not executed" have real output.

(exit-criteria)=
### Exit criteria

Section 9 gives the same gate as ordered steps with commands, owners and pass criteria; where the two differ, section 9 is current.

All of the following must be produced as actual command output on a machine with `torch` installed:

1. `uv sync` succeeds.
2. `uv run pytest tests/unit/model_bridge/supported_architectures/test_cohere_adapter.py -v` passes.
3. `uv run pytest tests/integration/model_bridge/test_cohere_adapter.py -v` passes, including the offline `d_head = 8` parity classes with and without QK-norm.
4. `make test-pr` passes, covering every adapter that shares the attention bridge.
5. `make check-format` and `uv run mypy .` pass with the pinned versions.
6. `verify_models --reverify --architectures CohereForCausalLM` passes Phase 1 at 100% on at least one checkpoint with `d_head > 2`, and on a `use_qk_norm=True` checkpoint before Command-R+ is advertised.

Until then, treat every claim in section 1 as a hypothesis supported by source reading only.
