#!/usr/bin/env python3
"""
bench_android.py - Parse llama-bench JSON output into normalized JSONL.

Reads the JSON output from llama-bench (Android) and produces schema-compatible
JSONL rows for the prefill and decode phases.
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any


def parse_model_metadata_from_filename(filename: str) -> dict[str, Any]:
    """Parse model metadata from llama-bench model filename."""
    # llama-bench outputs "model_filename": "../model.gguf"
    # Extract what we can from the path if actual loaded metadata isn't available
    return {
        "model_name_log": None,
        "model_type_log": None,
        "model_params_log": None,
        "file_size_log": None,
    }


def build_row(
    bench_row: dict[str, Any],
    run_dir: Path,
    model_id: str,
    model_checksum: str,
    quant_format: str,
    device_serial: str,
    device_model: str,
    threads: int,
    cpu_mask: str,
    run_count: int,
    ctx: int,
    llama_cpp_commit: str,
    model_path: str,
) -> dict[str, Any]:
    """Build a normalized JSON row from a single llama-bench result row."""

    n_prompt = bench_row.get("n_prompt", 0)
    n_gen = bench_row.get("n_gen", 0)
    avg_ns = bench_row.get("avg_ns", 0)
    stddev_ns = bench_row.get("stddev_ns", 0)
    samples_ns = bench_row.get("samples_ns", [])

    # Determine test type: prefill if n_prompt > 0, decode if n_gen > 0
    is_prefill = n_prompt > 0 and n_gen == 0
    is_decode = n_gen > 0 and n_prompt == 0

    # llama-bench reports nanoseconds per sample
    # avg_ts is tokens per second (throughput, not per-token latency)
    avg_ts = bench_row.get("avg_ts", 0)  # tokens per second
    stddev_ts = bench_row.get("stddev_ts", 0)

    if is_decode and n_gen > 0:
        # Decode: avg_ns is total time for generating n_gen tokens
        # avg_ts is throughput (tok/s), not per-token latency
        median_ns = statistics.median(samples_ns) if samples_ns else avg_ns
        elapsed_ms = median_ns / 1e6
        # For decode, ttft is the total elapsed time (no separate TTFT concept in decode-only)
        ttft_ms = elapsed_ms
        decode_tok_per_sec = avg_ts
        prefill_tok_per_sec = None
        prompt_tokens = None
        generated_tokens = n_gen
    elif is_prefill and n_prompt > 0:
        # Prefill: total time to process n_prompt tokens
        median_ns = statistics.median(samples_ns) if samples_ns else avg_ns
        elapsed_ms = median_ns / 1e6
        ttft_ms = elapsed_ms  # total prefill time is the TTFT equivalent
        prefill_tok_per_sec = avg_ts if avg_ts > 0 else None
        decode_tok_per_sec = None
        prompt_tokens = n_prompt
        generated_tokens = 0
    else:
        elapsed_ms = avg_ns / 1e6
        ttft_ms = elapsed_ms
        decode_tok_per_sec = avg_ts if is_decode else None
        prefill_tok_per_sec = avg_ts if is_prefill else None
        prompt_tokens = n_prompt if n_prompt > 0 else None
        generated_tokens = n_gen if n_gen > 0 else None

    # llama-bench model metadata
    model_type = bench_row.get("model_type", "")
    model_size = bench_row.get("model_size", 0)
    model_n_params = bench_row.get("model_n_params", 0)

    # Build filename-based metadata as fallback
    model_name_log = bench_row.get("model_filename", "")
    model_params_log = f"{model_n_params / 1e9:.2f} B" if model_n_params > 0 else None
    file_size_log = f"{model_size / (1024**3):.2f} GiB" if model_size > 0 else None

    row = {
        "schema_version": 1,
        "backend": "llama_cpp_cpu",
        "model_id": model_id,
        "model_checksum": model_checksum,
        "packed_checksum": None,
        "quant_format": quant_format,
        "prompt_id": "llama-bench",
        "seed": 0,
        "context_length_cap": ctx,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "load_time_ms": None,
        "ttft_ms": round(ttft_ms, 2) if ttft_ms else None,
        "prefill_tok_per_sec": round(prefill_tok_per_sec, 2) if prefill_tok_per_sec else None,
        "decode_tok_per_sec": round(decode_tok_per_sec, 2) if decode_tok_per_sec else None,
        "elapsed_ms": round(elapsed_ms, 2) if elapsed_ms else None,
        "warmup_count": 0,
        "run_count": run_count,
        "timing_method": "median" if len(samples_ns) > 1 else "single",
        "device_serial": device_serial,
        "device_model": device_model,
        "threads": threads,
        "cpu_mask": cpu_mask,
        "llama_cpp_commit": llama_cpp_commit,
        "model_path": model_path,
        "run_dir": str(run_dir),
        "model_name_log": model_name_log,
        "model_type_log": model_type,
        "model_params_log": model_params_log,
        "file_size_log": file_size_log,
        "status": "ok",
        "error": None,
        "raw_logs": None,
    }

    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse llama-bench JSON output into normalized JSONL."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to run directory containing llama_bench_output.json",
    )
    parser.add_argument("--model-id", required=True, help="Model identifier")
    parser.add_argument("--model-checksum", required=True, help="SHA256 checksum of model file")
    parser.add_argument("--quant-format", required=True, help="Quantization format")
    parser.add_argument("--device-serial", required=True, help="ADB device serial")
    parser.add_argument("--device-model", required=True, help="Device model string")
    parser.add_argument("--threads", type=int, required=True, help="Number of threads")
    parser.add_argument("--cpu-mask", required=True, help="CPU mask used")
    parser.add_argument("--run-count", type=int, required=True, help="Number of runs")
    parser.add_argument("--ctx", type=int, required=True, help="Context length")
    parser.add_argument("--llama-cpp-commit", required=True, help="llama.cpp git commit")
    parser.add_argument("--model-path", required=True, help="Path to model on device")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL file path")

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    json_file = run_dir / "llama_bench_output.json"

    if not json_file.exists():
        print(f"Error: llama-bench output not found: {json_file}", file=sys.stderr)
        sys.exit(1)

    with open(json_file) as f:
        bench_data = json.load(f)

    if not isinstance(bench_data, list):
        print(f"Error: Expected JSON array from llama-bench, got {type(bench_data)}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a") as out_f:
        for bench_row in bench_data:
            try:
                row = build_row(
                    bench_row=bench_row,
                    run_dir=run_dir,
                    model_id=args.model_id,
                    model_checksum=args.model_checksum,
                    quant_format=args.quant_format,
                    device_serial=args.device_serial,
                    device_model=args.device_model,
                    threads=args.threads,
                    cpu_mask=args.cpu_mask,
                    run_count=args.run_count,
                    ctx=args.ctx,
                    llama_cpp_commit=args.llama_cpp_commit,
                    model_path=args.model_path,
                )
                out_f.write(json.dumps(row) + "\n")
            except Exception as e:
                error_row = {
                    "schema_version": 1,
                    "backend": "llama_cpp_cpu",
                    "model_id": args.model_id,
                    "prompt_id": "llama-bench",
                    "status": "failed",
                    "error": str(e),
                    "run_dir": str(run_dir),
                    "model_name_log": None,
                    "model_type_log": None,
                    "model_params_log": None,
                    "file_size_log": None,
                }
                out_f.write(json.dumps(error_row) + "\n")


if __name__ == "__main__":
    main()
