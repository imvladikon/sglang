"""CPU contract tests for GLM-5.3-Flash hybrid-attention LoRA."""

from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

from sglang.srt.configs.glm5_next import Glm5NextTextConfig
from sglang.srt.lora.utils import (
    _KNOWN_LORA_TARGET_MODULES,
    ATTN_TP_LORA_MODULE_NAMES,
    KDA_GATE_LORA_NAMES,
    REPLICATED_LINEAR_LORA_NAMES,
    _model_declares_scoped_lora_target,
    get_default_hidden_dim,
)
from sglang.srt.models.glm5_next import (
    Glm5NextForConditionalGeneration,
    _can_fuse_kda_projections,
)
from sglang.srt.utils.common import SUPPORTED_LORA_TARGET_MODULES


def _fake_model(*, layers=45, experts=288):
    kda_layers = [index for index in range(layers) if index % 4 != 3]
    config = Glm5NextTextConfig(
        hidden_size=4096,
        intermediate_size=12288,
        moe_intermediate_size=2048,
        num_hidden_layers=layers,
        num_attention_heads=64,
        num_key_value_heads=64,
        n_routed_experts=experts,
        n_shared_experts=1,
        num_experts_per_tok=8,
        first_k_dense_replace=3,
        # Match the released full and surgery configs exactly.
        moe_layer_freq=None,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        v_head_dim=256,
        linear_attn_config={
            "num_heads": 64,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "kda_layers": kda_layers,
            "full_attn_layers": [
                index for index in range(layers) if index not in kda_layers
            ],
        },
    )
    model = Glm5NextForConditionalGeneration.__new__(Glm5NextForConditionalGeneration)
    model.config = config
    return model


def test_full_and_9b_layer_counts_match_the_contract():
    full = _fake_model()
    surgery = _fake_model(layers=10, experts=32)

    assert len(full.config.linear_layer_ids) == 34
    assert len(full.config.full_attention_layer_ids) == 11
    assert len(surgery.config.linear_layer_ids) == 8
    assert len(surgery.config.full_attention_layer_ids) == 2


def test_kda_and_dsa_attention_geometry_is_layer_aware():
    model = _fake_model()

    assert model.get_hidden_dim("qkv_proj", 0) == (4096, 24576)
    assert model.get_hidden_dim("o_proj", 0) == (8192, 4096)
    assert model.get_hidden_dim("b_proj", 0) == (4096, 64)
    assert model.get_hidden_dim("f_a_proj", 0) == (4096, 128)
    assert model.get_hidden_dim("f_b_proj", 0) == (128, 8192)
    assert model.get_hidden_dim("g_a_proj", 0) == (4096, 128)
    assert model.get_hidden_dim("g_b_proj", 0) == (128, 8192)

    assert model.get_hidden_dim("fused_qkv_a_proj_with_mqa", 3) == (4096, 2048)
    assert model.get_hidden_dim("q_b_proj", 3) == (1536, 16384)
    assert model.get_hidden_dim("kv_b_proj", 3) == (512, 32768)
    assert model.get_hidden_dim("o_proj", 3) == (16384, 4096)


def test_dense_shared_and_routed_expert_geometry_accepts_null_frequency():
    model = _fake_model()

    assert model.get_hidden_dim("gate_up_proj", 2) == (4096, 24576)
    assert model.get_hidden_dim("down_proj", 2) == (12288, 4096)
    assert model.get_hidden_dim("gate_up_proj", 3) == (4096, 4096)
    assert model.get_hidden_dim("down_proj", 3) == (2048, 4096)
    assert model.get_hidden_dim("gate_up_proj_moe", 3) == (4096, 4096)
    assert model.get_hidden_dim("down_proj_moe", 3) == (2048, 4096)


def test_kda_gate_registry_and_parallelism_contract():
    model = _fake_model()

    assert KDA_GATE_LORA_NAMES <= _KNOWN_LORA_TARGET_MODULES
    assert KDA_GATE_LORA_NAMES <= set(SUPPORTED_LORA_TARGET_MODULES)
    assert KDA_GATE_LORA_NAMES <= set(model.supported_lora_modules)
    assert {"f_a_proj", "g_a_proj"} <= set(REPLICATED_LINEAR_LORA_NAMES)
    assert {"b_proj", "f_b_proj", "g_b_proj"} <= ATTN_TP_LORA_MODULE_NAMES
    assert get_default_hidden_dim("g_b_proj", model.config, 0) == (128, 8192)


def test_kda_gate_names_do_not_leak_into_other_models_all_target():
    generic = SimpleNamespace(supported_lora_modules=("qkv_proj",))
    glm = SimpleNamespace(
        supported_lora_modules=Glm5NextForConditionalGeneration.supported_lora_modules
    )

    assert not _model_declares_scoped_lora_target(generic, "b_proj")
    assert _model_declares_scoped_lora_target(glm, "b_proj")
    assert _model_declares_scoped_lora_target(generic, "qkv_proj")


def test_lora_disables_the_unquantized_fused_kda_layout():
    with patch(
        "sglang.srt.models.glm5_next.get_lora",
        return_value=SimpleNamespace(enable_lora=True),
    ):
        assert not _can_fuse_kda_projections(None, 2, 2)

    with patch(
        "sglang.srt.models.glm5_next.get_lora",
        return_value=SimpleNamespace(enable_lora=False),
    ):
        assert _can_fuse_kda_projections(None, 2, 2)
        assert not _can_fuse_kda_projections(None, 1, 2)
        assert not _can_fuse_kda_projections(object(), 2, 2)
