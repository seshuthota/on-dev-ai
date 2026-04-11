#!/usr/bin/env python3
"""
normalize_llama_cli.py - Parse llama-cli output logs into normalized JSONL.

Reads run logs from a benchmark run directory and produces schema-compatible
JSONL rows summarizing the measured runs.
"""

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Optional


def parse_perf_lines(log_content: str) -> dict[str, Any]:
    """Parse llama-cli performance lines from log content."""
    result = {
        "load_time_ms": None,
        "prompt_eval_ms": None,
        "prompt_tokens": None,
        "prompt_tok_per_sec": None,
        "eval_ms": None,
        "eval_runs": None,
        "eval_tok_per_sec": None,
        "total_time_ms": None,
        "total_tokens": None,
        "generated_text": None,
    }

    for line in log_content.splitlines():
        line = line.strip()

        m = re.search(r"load time\s*=\s*([0-9.]+)\s*ms", line)
        if m:
            result["load_time_ms"] = float(m.group(1))

        m = re.search(
            r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens\s*\(\s*([0-9.]+)\s*ms per token,\s*([0-9.]+)\s*tokens per second\)",
            line,
        )
        if m:
            result["prompt_eval_ms"] = float(m.group(1))
            result["prompt_tokens"] = int(m.group(2))
            result["prompt_tok_per_sec"] = float(m.group(4))

        m = re.search(
            r"eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*runs?\s*\(\s*([0-9.]+)\s*ms per token,\s*([0-9.]+)\s*tokens per second\)",
            line,
        )
        if m:
            result["eval_ms"] = float(m.group(1))
            result["eval_runs"] = int(m.group(2))
            result["eval_tok_per_sec"] = float(m.group(4))

        m = re.search(
            r"total time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens",
            line,
        )
        if m:
            result["total_time_ms"] = float(m.group(1))
            result["total_tokens"] = int(m.group(2))

    return result


def parse_generated_tokens(log_content: str) -> int:
    """Extract generated token count from log content."""
    lines = log_content.splitlines()
    for line in reversed(lines):
        line = line.strip()
        m = re.search(r"\[([0-9]+)\/([0-9]+)\]\s*$", line)
        if m:
            return int(m.group(2))
    return 0


def compute_median(values: list[float]) -> Optional[float]:
    """Compute median of a list of values."""
    if not values:
        return None
    return statistics.median(values)


def build_row(
    run_dir: Path,
    prompt_id: str,
    model_id: str,
    model_checksum: str,
    quant_format: str,
    device_serial: str,
    device_model: str,
    threads: int,
    cpu_mask: str,
    warmup_count: int,
    run_count: int,
    ctx: int,
    install_dir: str,
    llama_cpp_commit: str,
    model_path: str,
) -> dict[str, Any]:
    """Build a normalized JSON row for a prompt_id across all runs."""

    row = {
        "schema_version": 1,
        "backend": "llama_cpp_cpu",
        "model_id": model_id,
        "model_checksum": model_checksum,
        "packed_checksum": None,
        "quant_format": quant_format,
        "prompt_id": prompt_id,
        "seed": 0,
        "context_length_cap": ctx,
        "prompt_tokens": None,
        "generated_tokens": None,
        "load_time_ms": None,
        "ttft_ms": None,
        "prefill_tok_per_sec": None,
        "decode_tok_per_sec": None,
        "elapsed_ms": None,
        "warmup_count": warmup_count,
        "run_count": run_count,
        "timing_method": "median",
        "device_serial": device_serial,
        "device_model": device_model,
        "threads": threads,
        "cpu_mask": cpu_mask,
        "llama_cpp_commit": llama_cpp_commit,
        "model_path": model_path,
        "status": "ok",
        "error": None,
        "raw_logs": None,
    }

    completed_file = run_dir / "completed_prompts.txt"
    if not completed_file.exists():
        row["status"] = "failed"
        row["error"] = "completed_prompts.txt not found"
        return row

    completed = completed_file.read_text().splitlines()
    if prompt_id not in completed:
        row["status"] = "failed"
        row["error"] = f"prompt_id {prompt_id} not in completed list"
        return row

    measured_logs = []
    for i in range(1, run_count + 1):
        log_file = run_dir / f"{prompt_id}_measured_{i}.log"
        if log_file.exists():
            measured_logs.append(log_file.read_text())

    if not measured_logs:
        row["status"] = "failed"
        row["error"] = f"No measured logs found for {prompt_id}"
        return row

    all_parsed = []
    for log_content in measured_logs:
        parsed = parse_perf_lines(log_content)
        gen_tokens = parse_generated_tokens(log_content)
        if gen_tokens == 0 and parsed["eval_runs"] is not None:
            gen_tokens = parsed["eval_runs"]
        parsed["generated_tokens"] = gen_tokens
        all_parsed.append(parsed)

    prompt_eval_times = [
        p["prompt_eval_ms"] for p in all_parsed if p["prompt_eval_ms"] is not None
    ]
    eval_times = [p["eval_ms"] for p in all_parsed if p["eval_ms"] is not None]
    prefill_tok_per_sec_values = [
        p["prompt_tok_per_sec"]
        for p in all_parsed
        if p["prompt_tok_per_sec"] is not None
    ]
    decode_tok_per_sec_values = [
        p["eval_tok_per_sec"] for p in all_parsed if p["eval_tok_per_sec"] is not None
    ]
    load_times = [
        p["load_time_ms"] for p in all_parsed if p["load_time_ms"] is not None
    ]
    generated_tokens_list = [p["generated_tokens"] for p in all_parsed]

    total_time_ms_values = [
        p["total_time_ms"] for p in all_parsed if p["total_time_ms"] is not None
    ]

    if not decode_tok_per_sec_values:
        row["status"] = "failed"
        row["error"] = f"No decode metrics found in measured logs for {prompt_id}"
        return row

    row["prompt_tokens"] = all_parsed[0]["prompt_tokens"]
    row["load_time_ms"] = compute_median(load_times) if load_times else None
    row["prefill_tok_per_sec"] = (
        compute_median(prefill_tok_per_sec_values)
        if prefill_tok_per_sec_values
        else None
    )
    row["decode_tok_per_sec"] = (
        compute_median(decode_tok_per_sec_values) if decode_tok_per_sec_values else None
    )
    row["generated_tokens"] = (
        compute_median(generated_tokens_list) if generated_tokens_list else None
    )

    median_prompt_eval = (
        compute_median(prompt_eval_times) if prompt_eval_times else None
    )
    median_eval = compute_median(eval_times) if eval_times else None
    median_total_time = (
        compute_median(total_time_ms_values) if total_time_ms_values else None
    )

    if median_prompt_eval is not None:
        row["ttft_ms"] = median_prompt_eval
    elif median_eval is not None:
        row["ttft_ms"] = median_eval

    if median_total_time is not None:
        row["elapsed_ms"] = median_total_time
    elif median_prompt_eval is not None and median_eval is not None:
        row["elapsed_ms"] = median_prompt_eval + median_eval
    elif median_eval is not None:
        row["elapsed_ms"] = median_eval

    raw_log = "\n".join(
        f"[run {i + 1}]\n{log}" for i, log in enumerate(measured_logs[:2])
    )
    row["raw_logs"] = raw_log[:10000]

    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse llama-cli output logs into normalized JSONL."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to run directory containing log files",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Model identifier",
    )
    parser.add_argument(
        "--model-checksum",
        required=True,
        help="SHA256 checksum of model file",
    )
    parser.add_argument(
        "--quant-format",
        required=True,
        help="Quantization format (e.g., Q4_K_M)",
    )
    parser.add_argument(
        "--device-serial",
        required=True,
        help="ADB device serial",
    )
    parser.add_argument(
        "--device-model",
        required=True,
        help="Device model string",
    )
    parser.add_argument(
        "--threads",
        type=int,
        required=True,
        help="Number of threads used",
    )
    parser.add_argument(
        "--cpu-mask",
        required=True,
        help="CPU mask used",
    )
    parser.add_argument(
        "--warmup-count",
        type=int,
        required=True,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--run-count",
        type=int,
        required=True,
        help="Number of measured runs",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        required=True,
        help="Context length",
    )
    parser.add_argument(
        "--install-dir",
        required=True,
        help="Path to install directory",
    )
    parser.add_argument(
        "--llama-cpp-commit",
        required=True,
        help="llama.cpp git commit hash",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to model on device",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Output JSONL file path",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    prompts_file = (
        run_dir.parent.parent.parent / "docs" / "custom_runtime_benchmark_prompts.jsonl"
    )
    prompt_ids = []
    if prompts_file.exists():
        with open(prompts_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    pid = obj.get("prompt_id")
                    if pid:
                        prompt_ids.append(pid)
                except json.JSONDecodeError:
                    continue

    if not prompt_ids:
        completed_file = run_dir / "completed_prompts.txt"
        if completed_file.exists():
            prompt_ids = completed_file.read_text().splitlines()

    if not prompt_ids:
        print("Error: Could not determine prompt IDs", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a") as out_f:
        for prompt_id in prompt_ids:
            try:
                row = build_row(
                    run_dir=run_dir,
                    prompt_id=prompt_id,
                    model_id=args.model_id,
                    model_checksum=args.model_checksum,
                    quant_format=args.quant_format,
                    device_serial=args.device_serial,
                    device_model=args.device_model,
                    threads=args.threads,
                    cpu_mask=args.cpu_mask,
                    warmup_count=args.warmup_count,
                    run_count=args.run_count,
                    ctx=args.ctx,
                    install_dir=args.install_dir,
                    llama_cpp_commit=args.llama_cpp_commit,
                    model_path=args.model_path,
                )
                out_f.write(json.dumps(row) + "\n")
            except Exception as e:
                error_row = {
                    "schema_version": 1,
                    "backend": "llama_cpp_cpu",
                    "model_id": args.model_id,
                    "prompt_id": prompt_id,
                    "status": "failed",
                    "error": str(e),
                }
                out_f.write(json.dumps(error_row) + "\n")


if __name__ == "__main__":
    main()
