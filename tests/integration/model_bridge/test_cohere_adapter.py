"""Integration tests for Cohere architecture adapter (CohereForCausalLM).

Model: trl-internal-testing/tiny-CohereForCausalLM
  - 2 layers, ~8M params, CPU-safe, no gating required
  - tie_word_embeddings=True by default
  - logit_scale=0.125 (canonical Command-R is 0.0625; tiny diverges so
    regression tests catch silent-fallback bugs in the passthrough)

NOTE: The Hub tiny model cannot validate attention on its own. It has d_head=2,
where Cohere's adjacent-pair RoPE and Llama's half-split RoPE are the same
rotation, and use_qk_norm=False. PositionEmbeddingsAttentionBridge reimplements
attention (it does not call CohereAttention.forward), so both paths are covered
by the offline classes at the bottom of this file, which build random
CohereForCausalLM models with d_head=8, GQA, and optional QK-norm.
"""

import copy
from typing import Any

import pytest
import torch
from transformers import AutoModelForCausalLM, CohereConfig, CohereForCausalLM

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import NormalizationBridge
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)

MODEL = "trl-internal-testing/tiny-CohereForCausalLM"


@pytest.fixture(scope="module")
def cohere_bridge():
    """Load tiny Cohere bridge once per module (no weight processing)."""
    return TransformerBridge.boot_transformers(MODEL, device="cpu")


@pytest.fixture(scope="module")
def cohere_bridge_processed():
    """Bridge with preprocess_weights applied (fold only, no centering).

    process_weights must be called explicitly — boot_transformers does not call
    it automatically. We disable all ProcessWeights options so only the adapter's
    preprocess_weights (logit_scale fold + untie) runs.
    """
    bridge = TransformerBridge.boot_transformers(MODEL, device="cpu")
    bridge.process_weights(
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    )
    return bridge


@pytest.fixture(scope="module")
def cohere_hf() -> Any:
    """Load the raw HF model for side-by-side comparisons."""
    return AutoModelForCausalLM.from_pretrained(MODEL).eval()


# ---------------------------------------------------------------------------
# 1. Bridge creation — exercises all 4 registration points + component_mapping
# ---------------------------------------------------------------------------


class TestCohereBridgeCreation:
    """Verify the bridge loads cleanly and exposes the expected structure."""

    def test_boot_transformers_succeeds(self, cohere_bridge: TransformerBridge) -> None:
        assert cohere_bridge is not None

    def test_block_count(self, cohere_bridge: TransformerBridge) -> None:
        # tiny model has 2 layers
        assert len(cohere_bridge.blocks) == 2

    def test_has_core_components(self, cohere_bridge: TransformerBridge) -> None:
        assert hasattr(cohere_bridge, "embed")
        assert hasattr(cohere_bridge, "unembed")
        assert hasattr(cohere_bridge, "ln_final")
        assert hasattr(cohere_bridge, "rotary_emb")

    def test_no_ln2_in_blocks(self, cohere_bridge: TransformerBridge) -> None:
        # Parallel block: no post_attention_layernorm
        for block in cohere_bridge.blocks:
            assert not hasattr(block, "ln2"), "Parallel block must not have ln2"

    def test_ln1_subtracts_mean_like_hf_layernorm(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        """uses_rms_norm=False must make ln1 mean-subtract (true LayerNorm), not RMS.

        Drives blocks[0].ln1 on a deliberately non-zero-mean input and compares
        against the raw HF CohereLayerNorm (which subtracts the mean). If
        uses_rms_norm regressed to True, the bridge would skip mean subtraction
        and diverge from HF by O(1), tripping this assertion.
        """
        ln1 = cohere_bridge.blocks[0].ln1
        hf_ln1 = cohere_hf.model.layers[0].input_layernorm
        d_model = cohere_bridge.cfg.d_model
        torch.manual_seed(0)
        # +5.0 shift guarantees a large mean so LN and RMS outputs differ markedly.
        x = torch.randn(1, 4, d_model) + 5.0
        with torch.no_grad():
            bridge_out = ln1(x)
            hf_out = hf_ln1(x)
        ln_diff = (bridge_out - hf_out).abs().max().item()
        assert ln_diff < 1e-4, f"ln1 does not match HF CohereLayerNorm: max_diff={ln_diff:.6f}"
        # And it must NOT match an RMS-only (no mean subtraction) normalization.
        variance = x.float().pow(2).mean(-1, keepdim=True)
        rms_only = (
            x.float()
            * torch.rsqrt(variance + ln1.original_component.variance_epsilon)
            * ln1.original_component.weight.float()
        ).to(bridge_out.dtype)
        rms_diff = (bridge_out - rms_only).abs().max().item()
        assert rms_diff > 1e-2, (
            "ln1 output matches RMS-only normalization; uses_rms_norm must be False "
            f"so the mean is subtracted (rms_diff={rms_diff:.6f})"
        )

    def test_cfg_logit_scale_matches_hf(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        """Regression: logit_scale must propagate from HF (not silently fall back to 0.0625)."""
        bridge_scale = getattr(cohere_bridge.cfg, "logit_scale")
        assert bridge_scale == cohere_hf.config.logit_scale
        # Anchor 0.125 so a passthrough regression that defaults to 0.0625 also trips here.
        assert bridge_scale == pytest.approx(0.125)

    def test_cfg_rope_parameters_matches_hf(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        """Regression: rope_parameters must propagate from HF (same passthrough trap as logit_scale)."""
        assert getattr(cohere_bridge.cfg, "rope_parameters") == cohere_hf.config.rope_parameters


# ---------------------------------------------------------------------------
# 2. Forward equivalence — HF logits ≈ bridge logits
# ---------------------------------------------------------------------------


class TestCohereForwardEquivalence:
    """Verify bridge and HF produce identical logits for the same input."""

    def test_forward_returns_logits(self, cohere_bridge: TransformerBridge) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            output = cohere_bridge(tokens)
        assert output.shape[0] == 1
        assert output.shape[1] == 4
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_matches_hf(self, cohere_bridge: TransformerBridge, cohere_hf: Any) -> None:
        """Unprocessed bridge logits must match HF."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            bridge_out = cohere_bridge(tokens)
            hf_out = cohere_hf(tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff:.6f}"


# ---------------------------------------------------------------------------
# 3. Logit scale applied end-to-end
# ---------------------------------------------------------------------------


class TestCohereLogitScaleEndToEnd:
    """Verify logit_scale is correctly folded into the loaded model.

    process_weights must be called before the fold takes effect — boot_transformers
    alone does NOT call process_weights. cohere_bridge_processed uses a fixture that
    calls process_weights with all standard options disabled so only preprocess_weights
    (the logit_scale fold) runs.
    """

    def test_unembed_weight_is_scaled_relative_to_hf(
        self, cohere_bridge_processed: TransformerBridge, cohere_hf: Any
    ) -> None:
        # After preprocess_weights, lm_head.weight inside the bridge should equal
        # HF lm_head.weight * logit_scale (both [d_vocab, d_model]).
        logit_scale = getattr(cohere_bridge_processed.cfg, "logit_scale")
        tl_weight = cohere_bridge_processed.unembed.original_component.weight  # [d_vocab, d_model]
        hf_weight = cohere_hf.lm_head.weight.detach()  # [d_vocab, d_model]
        expected = hf_weight * logit_scale
        max_diff = (tl_weight - expected).abs().max().item()
        assert max_diff < 1e-5, (
            f"unembed.weight not correctly scaled: max_diff={max_diff:.6f}, "
            f"logit_scale={logit_scale}"
        )

    def test_logit_scale_not_folded_before_process_weights(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        """Before process_weights only HF's runtime multiply applies the scale."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            bridge_out = cohere_bridge(tokens)
            hf_out = cohere_hf(tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Possible double-application of logit_scale; diff={max_diff:.6f}"

    def test_processed_forward_matches_hf(
        self, cohere_bridge_processed: TransformerBridge, cohere_hf: Any
    ) -> None:
        """Regression: after the fold, HF's runtime multiply must not apply the scale again.

        The forward still runs CohereForCausalLM.forward. If its logit_scale were
        left in place the processed logits would be logit_scale times too small.
        """
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            bridge_out = cohere_bridge_processed(tokens)
            hf_out = cohere_hf(tokens).logits
        assert torch.allclose(bridge_out, hf_out, rtol=1e-3, atol=1e-6), (
            "Processed logits diverge from HF (logit_scale applied twice?): "
            f"max_diff={(bridge_out - hf_out).abs().max().item():.6g}, "
            f"hf_max={hf_out.abs().max().item():.6g}"
        )


# ---------------------------------------------------------------------------
# 4. Tied embedding preserved — embed.W_E must NOT be scaled
# ---------------------------------------------------------------------------


class TestCohereTiedEmbedding:
    """Verify preprocess_weights does not corrupt embed.W_E in the tied case.

    Both tests use cohere_bridge_processed (process_weights called with fold-only
    options) because the untie + fold only happens inside process_weights.
    """

    def test_embed_weight_equals_hf_embed_tokens(
        self, cohere_bridge_processed: TransformerBridge, cohere_hf: Any
    ) -> None:
        # After the fold, embed.W_E must still equal HF's unscaled embed_tokens.weight.
        # If the fold corrupted embed (in-place on the shared tensor), this fails.
        hf_embed = cohere_hf.model.embed_tokens.weight.detach()  # [d_vocab, d_model]
        tl_embed = cohere_bridge_processed.embed.W_E  # [d_vocab, d_model]
        max_diff = (tl_embed - hf_embed).abs().max().item()
        assert (
            max_diff < 1e-6
        ), f"embed.W_E was corrupted (possibly by logit_scale fold): max_diff={max_diff:.6f}"

    @pytest.mark.parametrize("logit_scale", [0.0625, 1.0])
    def test_embed_and_unembed_weights_differ(self, logit_scale: float) -> None:
        # After the logit_scale fold, embed.W_E and unembed.weight must NOT be
        # identical for a non-trivial scale. logit_scale=1.0 is kept as a regression
        # guard for the no-op case, where the two weights stay tied.
        #
        # cfg.logit_scale is set before process_weights so the fold (which reads it
        # inside preprocess_weights) runs with the parametrized value.
        bridge = TransformerBridge.boot_transformers(MODEL, device="cpu")
        setattr(bridge.cfg, "logit_scale", logit_scale)
        bridge.process_weights(
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
            refactor_factored_attn_matrices=False,
        )
        tl_embed = bridge.embed.W_E
        tl_unembed = bridge.unembed.original_component.weight
        weights_identical = torch.allclose(tl_embed, tl_unembed)
        if logit_scale == 1.0:
            assert weights_identical, (
                "embed.W_E and unembed.weight should remain tied when logit_scale=1.0 "
                "(the fold is a no-op)"
            )
        else:
            assert not weights_identical, (
                "embed.W_E and unembed.weight are identical — "
                "logit_scale fold may not have been applied or untied correctly"
            )


# ---------------------------------------------------------------------------
# 5. HF delegation — RoPE, attention, normalization go through HF modules
# ---------------------------------------------------------------------------


class TestCohereHFDelegation:
    """Spot-check that bridge submodules delegate to actual HF modules.

    This confirms setup_component_testing wired rotary_emb correctly and that
    NormalizationBridge and PositionEmbeddingsAttentionBridge hold live HF objects.
    """

    def test_ln1_original_component_is_hf_norm(self, cohere_bridge: TransformerBridge) -> None:
        # bridge.blocks[0].ln1.original_component should be the live HF CohereLayerNorm
        ln1 = cohere_bridge.blocks[0].ln1
        assert isinstance(ln1, NormalizationBridge)
        assert ln1.original_component is not None
        # CohereLayerNorm has variance_epsilon (not eps)
        assert hasattr(
            ln1.original_component, "variance_epsilon"
        ), "original_component is not CohereLayerNorm (missing variance_epsilon)"

    def test_attn_original_component_is_hf_attention(
        self, cohere_bridge: TransformerBridge
    ) -> None:
        attn = cohere_bridge.blocks[0].attn
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.original_component is not None
        # CohereAttention has q_proj
        assert hasattr(
            attn.original_component, "q_proj"
        ), "original_component is not CohereAttention (missing q_proj)"

    def test_ln_final_original_component_is_hf_norm(self, cohere_bridge: TransformerBridge) -> None:
        assert cohere_bridge.ln_final.original_component is not None
        assert hasattr(cohere_bridge.ln_final.original_component, "variance_epsilon")


# ---------------------------------------------------------------------------
# 6. Parallel-attn hooks fire correctly
# ---------------------------------------------------------------------------


class TestCohereParallelHooks:
    """Verify hook placement for the parallel attention+MLP block."""

    def test_no_hook_resid_mid(self, cohere_bridge: TransformerBridge) -> None:
        # Parallel block has no intermediate residual stream
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = cohere_bridge.run_with_cache(tokens)
        assert not any("hook_resid_mid" in k for k in cache.keys())

    def test_attn_and_mlp_hooks_fire(self, cohere_bridge: TransformerBridge) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = cohere_bridge.run_with_cache(tokens)
        for i in range(2):
            assert f"blocks.{i}.attn.hook_in" in cache
            assert f"blocks.{i}.attn.hook_out" in cache
            assert f"blocks.{i}.mlp.hook_in" in cache
            assert f"blocks.{i}.mlp.hook_out" in cache

    def test_residual_hooks_fire(self, cohere_bridge: TransformerBridge) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = cohere_bridge.run_with_cache(tokens)
        for i in range(2):
            assert f"blocks.{i}.hook_resid_pre" in cache
            assert f"blocks.{i}.hook_resid_post" in cache


# ---------------------------------------------------------------------------
# 7. Offline parity on a non-degenerate model (d_head=8, GQA, optional QK-norm)
# ---------------------------------------------------------------------------

OFFLINE_TOKENS = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29]])


def _offline_cohere(use_qk_norm: bool) -> tuple[TransformerBridge, Any]:
    """Bridge + independent HF reference for a random CohereForCausalLM (no Hub access).

    Weights are re-drawn at a larger scale than HF's init so attention scores are
    O(1); with the default init attention is near-uniform and RoPE errors vanish.
    """
    torch.manual_seed(0)
    cfg = CohereConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        logit_scale=0.125,
        use_qk_norm=use_qk_norm,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attn_implementation="eager",
    )
    hf_reference = CohereForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, param in hf_reference.named_parameters():
            if "norm" in name:
                param.uniform_(0.5, 1.5)
            else:
                param.normal_(mean=0.0, std=0.2)
    hf_reference.config._attn_implementation = "eager"
    for layer in hf_reference.model.layers:
        layer.self_attn.config._attn_implementation = "eager"

    hf_model = CohereForCausalLM(copy.deepcopy(cfg)).eval()
    hf_model.load_state_dict(hf_reference.state_dict())
    bridge = build_bridge_from_module(
        hf_model,
        "CohereForCausalLM",
        hf_config=copy.deepcopy(cfg),
        tokenizer=None,
        device="cpu",
    ).eval()
    bridge.adapter.setup_component_testing(hf_model, bridge_model=bridge)
    return bridge, hf_reference


@pytest.fixture(scope="module", params=[False, True], ids=["no_qk_norm", "qk_norm"])
def offline_cohere(request: pytest.FixtureRequest) -> tuple[TransformerBridge, Any]:
    return _offline_cohere(use_qk_norm=request.param)


class TestCohereOfflineParity:
    """Logit and attention-pattern parity where the RoPE convention matters."""

    def test_fixture_exercises_rope_and_gqa(
        self, offline_cohere: tuple[TransformerBridge, Any]
    ) -> None:
        bridge, hf_reference = offline_cohere
        # d_head must exceed 2 or the two RoPE conventions are indistinguishable.
        assert bridge.cfg.d_head == 8
        assert bridge.cfg.n_key_value_heads == 2
        assert bridge.cfg.use_qk_norm == hf_reference.config.use_qk_norm

    def test_forward_matches_hf(self, offline_cohere: tuple[TransformerBridge, Any]) -> None:
        bridge, hf_reference = offline_cohere
        with torch.no_grad():
            bridge_out = bridge(OFFLINE_TOKENS)
            hf_out = hf_reference(OFFLINE_TOKENS).logits
        assert bridge_out.shape == hf_out.shape
        # Guard against a degenerate reference where any output would "match".
        assert hf_out.abs().max().item() > 1e-2
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff:.6f}"

    def test_attention_pattern_matches_hf(
        self, offline_cohere: tuple[TransformerBridge, Any]
    ) -> None:
        bridge, hf_reference = offline_cohere
        with torch.no_grad():
            _, cache = bridge.run_with_cache(OFFLINE_TOKENS)
            hf_attentions = hf_reference(OFFLINE_TOKENS, output_attentions=True).attentions
        for i, hf_pattern in enumerate(hf_attentions):
            pattern = cache[f"blocks.{i}.attn.hook_pattern"]
            max_diff = (pattern - hf_pattern).abs().max().item()
            assert max_diff < 1e-4, f"Layer {i} attention pattern max diff = {max_diff:.6f}"

    def test_qk_norm_hooks_fire_only_when_enabled(
        self, offline_cohere: tuple[TransformerBridge, Any]
    ) -> None:
        bridge, hf_reference = offline_cohere
        _, cache = bridge.run_with_cache(OFFLINE_TOKENS)
        fired = "blocks.0.attn.hook_q_normed" in cache
        assert fired == bool(hf_reference.config.use_qk_norm)

    def test_llama_rotation_diverges_from_hf(self) -> None:
        """The fixture must be able to detect the wrong RoPE convention.

        Forcing Llama's half-split rotation onto this model reproduces the original
        defect; if this stops diverging, the parity tests above have gone blind.
        """
        bridge, hf_reference = _offline_cohere(use_qk_norm=False)
        for block in bridge.blocks:
            block.attn._rotary_interleaved = False
        with torch.no_grad():
            bridge_out = bridge(OFFLINE_TOKENS)
            hf_out = hf_reference(OFFLINE_TOKENS).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff > 1e-3, f"Half-split RoPE unexpectedly matches HF (diff={max_diff:.6f})"


class TestCohereOfflineProcessedLogitScale:
    """logit_scale must be applied exactly once after process_weights."""

    def test_processed_forward_matches_hf(self) -> None:
        bridge, hf_reference = _offline_cohere(use_qk_norm=False)
        bridge.process_weights(
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
            refactor_factored_attn_matrices=False,
        )
        with torch.no_grad():
            bridge_out = bridge(OFFLINE_TOKENS)
            hf_out = hf_reference(OFFLINE_TOKENS).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Processed bridge vs HF max diff = {max_diff:.6f}"


# ---------------------------------------------------------------------------
# 8. Acceptance: real-checkpoint logit parity vs Hugging Face (max abs diff < 1e-4)
# ---------------------------------------------------------------------------

# Ungated real checkpoint with d_head=128 and logit_scale=0.0625. Unlike MODEL above
# (d_head=2), its logits depend on the RoPE convention, so parity here is meaningful.
REAL_MODEL = "hyper-accel/tiny-random-cohere"
PARITY_TOL = 1e-4


@pytest.fixture(scope="module")
def real_cohere_bridge() -> TransformerBridge:
    return TransformerBridge.boot_transformers(REAL_MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def real_cohere_hf() -> Any:
    """HF reference loaded independently of the bridge's wrapped instance."""
    return AutoModelForCausalLM.from_pretrained(
        REAL_MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()


def _real_tokens(hf_model: Any, seq_len: int) -> torch.Tensor:
    """BOS followed by fixed, spread-out in-vocab ids (no tokenizer dependence)."""
    bos = hf_model.config.bos_token_id
    ids = [bos] + [(1000 + 977 * i) % hf_model.config.vocab_size for i in range(seq_len - 1)]
    return torch.tensor([ids])


# Not in the CI model cache (~0.5 GB download), so opt-in: pytest -m slow.
@pytest.mark.slow
class TestCohereRealCheckpointLogitParity:
    """Command-R acceptance gate: bridge logits equal HF logits to 1e-4 max abs diff."""

    def test_checkpoint_is_not_degenerate(
        self, real_cohere_bridge: TransformerBridge, real_cohere_hf: Any
    ) -> None:
        # d_head == 2 would make the RoPE conventions indistinguishable.
        assert real_cohere_bridge.cfg.d_head > 2
        assert getattr(real_cohere_bridge.cfg, "logit_scale") == real_cohere_hf.config.logit_scale

    @pytest.mark.parametrize("seq_len", [4, 32])
    def test_logits_match_hf_within_tolerance(
        self, real_cohere_bridge: TransformerBridge, real_cohere_hf: Any, seq_len: int
    ) -> None:
        tokens = _real_tokens(real_cohere_hf, seq_len)
        with torch.no_grad():
            bridge_logits = real_cohere_bridge(tokens)
            hf_logits = real_cohere_hf(tokens).logits
        assert bridge_logits.shape == hf_logits.shape
        hf_scale = hf_logits.abs().max().item()
        # An absolute tolerance is vacuous if the logits themselves are that small.
        assert hf_scale > 100 * PARITY_TOL, f"HF logits too small to test parity: {hf_scale:.2e}"
        max_diff = (bridge_logits - hf_logits).abs().max().item()
        assert max_diff < PARITY_TOL, (
            f"{REAL_MODEL!r} bridge vs HF max abs diff = {max_diff:.2e} "
            f"(tolerance {PARITY_TOL:.0e}, HF logit scale {hf_scale:.2e}, seq_len={seq_len})"
        )
