#!/usr/bin/env python3
"""
Verify C++ layer-0 reference tensor dumps against Python (torch) oracle.

Loads TinyLlama from safetensors, runs the same one-token layer-0 forward pass,
and compares selected indices with the C++ JSON dump.

Usage:
    python verify_reference_tensors.py \
        --model-dir /home/curious/models/TinyLlama-1.1B-Chat-v1.0 \
        --dump /tmp/ondevai_day7_dump.json \
        --prompt Hello
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer


HIDDEN_INDICES = [0, 1, 2, 3, 31, 127, 511, 1024, 1536, 2047]
LOGITS_INDICES = [0, 1, 2, 13, 100, 1000, 15043, 22172, 29991, 31999]
HIDDEN_TOL = 5e-3
LOGITS_TOL = 2e-2


def packed_weight(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.float16).to(torch.float32)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    scale = (variance + eps).rsqrt()
    return weight * x * scale


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def load_tinyllama_safetensors(model_dir: Path):
    state_dict = load_file(model_dir / "model.safetensors")
    config = {
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "vocab_size": 32000,
        "rms_norm_eps": 1e-5,
    }
    return state_dict, config


def compute_layer0(
    state_dict: dict,
    config: dict,
    token_id: int,
    rope_theta: float = 10000.0,
):
    hidden_size = config["hidden_size"]
    num_attention_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    intermediate_size = config["intermediate_size"]
    rms_norm_eps = config["rms_norm_eps"]
    head_dim = hidden_size // num_attention_heads

    embed_tokens = packed_weight(state_dict["model.embed_tokens.weight"])
    input_layernorm_weight = packed_weight(
        state_dict["model.layers.0.input_layernorm.weight"]
    )
    q_proj_weight = packed_weight(state_dict["model.layers.0.self_attn.q_proj.weight"])
    k_proj_weight = packed_weight(state_dict["model.layers.0.self_attn.k_proj.weight"])
    v_proj_weight = packed_weight(state_dict["model.layers.0.self_attn.v_proj.weight"])
    o_proj_weight = packed_weight(state_dict["model.layers.0.self_attn.o_proj.weight"])
    post_attn_layernorm_weight = packed_weight(
        state_dict["model.layers.0.post_attention_layernorm.weight"]
    )
    gate_proj_weight = packed_weight(state_dict["model.layers.0.mlp.gate_proj.weight"])
    up_proj_weight = packed_weight(state_dict["model.layers.0.mlp.up_proj.weight"])
    down_proj_weight = packed_weight(state_dict["model.layers.0.mlp.down_proj.weight"])
    final_norm_weight = packed_weight(state_dict["model.norm.weight"])
    lm_head_weight = packed_weight(state_dict["lm_head.weight"])

    x = embed_tokens[token_id].to(torch.float32)

    norm1_out = rmsnorm(x, input_layernorm_weight, rms_norm_eps)

    q = torch.matmul(q_proj_weight, norm1_out)
    k = torch.matmul(k_proj_weight, norm1_out)
    v = torch.matmul(v_proj_weight, norm1_out)

    q = q.view(num_attention_heads, head_dim)
    k = k.view(num_kv_heads, head_dim)
    v = v.view(num_kv_heads, head_dim)

    def rope_impl(
        x: torch.Tensor, position: int, rope_theta: float, head_dim: int
    ) -> torch.Tensor:
        num_heads = x.shape[0]
        device = x.device
        half_dim = head_dim // 2

        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, half_dim, device=device).float() / half_dim)
        )
        angle = position * inv_freq
        cos = torch.cos(angle)
        sin = torch.sin(angle)

        x0 = x[..., :half_dim]
        x1 = x[..., half_dim:]
        x0_new = x0 * cos - x1 * sin
        x1_new = x0 * sin + x1 * cos
        return torch.cat([x0_new, x1_new], dim=-1)

    q = rope_impl(q, 0, rope_theta, head_dim)
    k = rope_impl(k, 0, rope_theta, head_dim)

    q_heads_per_kv = num_attention_heads // num_kv_heads
    context = torch.zeros_like(q)

    for qh in range(num_attention_heads):
        kv_head = qh // q_heads_per_kv
        context[qh] = v[kv_head]

    attn_out = torch.matmul(o_proj_weight, context.flatten())

    residual1 = x + attn_out

    norm2_out = rmsnorm(residual1, post_attn_layernorm_weight, rms_norm_eps)

    gate = torch.matmul(gate_proj_weight, norm2_out)
    up = torch.matmul(up_proj_weight, norm2_out)
    mlp_intermediate = silu(gate) * up
    mlp_out = torch.matmul(down_proj_weight, mlp_intermediate)

    layer_out = residual1 + mlp_out

    final_norm_out = rmsnorm(layer_out, final_norm_weight, rms_norm_eps)

    logits = torch.matmul(lm_head_weight, final_norm_out)

    return {
        "embedding": x.numpy(),
        "rmsnorm": norm1_out.numpy(),
        "q_projection": q.flatten().numpy(),
        "attention_output": attn_out.numpy(),
        "mlp_output": mlp_out.numpy(),
        "final_logits": logits.numpy(),
    }


def verify_tensor(
    name: str, cpp_values: list, py_values: list, indices: list, tolerance: float
):
    max_diff = 0.0
    for idx in indices:
        if idx >= len(cpp_values) or idx >= len(py_values):
            print(
                f"  [WARN] index {idx} out of range ({len(cpp_values)}, {len(py_values)})"
            )
            continue
        diff = abs(cpp_values[idx] - py_values[idx])
        max_diff = max(max_diff, diff)

    status = "PASS" if max_diff <= tolerance else "FAIL"
    print(f"  {name}: {status} (max_diff={max_diff:.6e}, tolerance={tolerance:.1e})")
    return max_diff <= tolerance, max_diff


def main():
    parser = argparse.ArgumentParser(
        description="Verify C++ tensor dumps against Python oracle"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Path to TinyLlama model directory",
    )
    parser.add_argument(
        "--dump", required=True, type=Path, help="Path to C++ JSON dump"
    )
    parser.add_argument("--prompt", required=True, type=str, help="Prompt used in dump")
    args = parser.parse_args()

    print(f"Loading C++ dump from: {args.dump}")
    with open(args.dump) as f:
        dump = json.load(f)

    print(f"Loading tokenizer from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir), trust_remote_code=True
    )

    print(f"Encoding prompt: '{args.prompt}'")
    tokens = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not tokens:
        print("ERROR: tokenizer produced no tokens")
        return 1
    token_id = tokens[0]
    print(f"First token: {token_id}")

    if dump.get("prompt") != args.prompt:
        print(
            f"WARNING: dump prompt '{dump.get('prompt')}' != args.prompt '{args.prompt}'"
        )
    if dump.get("token_id") != token_id:
        print(f"ERROR: dump token_id {dump.get('token_id')} != computed {token_id}")
        return 1

    print("Loading TinyLlama safetensors...")
    state_dict, config = load_tinyllama_safetensors(args.model_dir)

    print("Computing layer 0 forward pass...")
    computed = compute_layer0(state_dict, config, token_id)

    print("\nVerifying tensors:")
    all_pass = True

    for key in [
        "embedding",
        "rmsnorm",
        "q_projection",
        "attention_output",
        "mlp_output",
    ]:
        if key not in dump:
            print(f"  {key}: MISSING in dump")
            all_pass = False
            continue
        cpp_vals = dump[key]["values"]
        py_vals = computed[key].tolist()
        indices = HIDDEN_INDICES if key != "q_projection" else HIDDEN_INDICES
        is_same_dim = len(cpp_vals) == len(py_vals)
        if not is_same_dim:
            print(f"  {key}: DIM MISMATCH cpp={len(cpp_vals)} py={len(py_vals)}")
            all_pass = False
            continue
        passed, _ = verify_tensor(key, cpp_vals, py_vals, indices, HIDDEN_TOL)
        all_pass = all_pass and passed

    if "final_logits" in dump:
        cpp_vals = dump["final_logits"]["values"]
        py_vals = computed["final_logits"].tolist()
        passed, max_diff = verify_tensor(
            "final_logits", cpp_vals, py_vals, LOGITS_INDICES, LOGITS_TOL
        )
        all_pass = all_pass and passed
        print(f"  logits max_diff: {max_diff:.6e}")

    print()
    if all_pass:
        print("RESULT: ALL CHECKS PASSED")
        return 0
    else:
        print("RESULT: SOME CHECKS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
