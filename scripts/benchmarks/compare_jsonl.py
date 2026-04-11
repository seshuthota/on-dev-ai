#!/usr/bin/env python3
"""
compare_jsonl.py - Compare benchmark results from multiple JSONL files.

Reads one or more JSONL files and prints a compact markdown table
grouped by backend, quant_format, model_id, prompt_id, threads, cpu_mask.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file, skipping malformed lines with a warning."""
    rows = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"Warning: Skipping malformed line {line_num} in {path}: {e}",
                    file=sys.stderr,
                )
    return rows


def group_key(row: dict[str, Any]) -> tuple:
    """Extract grouping key from a row."""
    return (
        row.get("backend", "unknown"),
        row.get("quant_format", "unknown"),
        row.get("model_id", "unknown"),
        row.get("prompt_id", "unknown"),
        row.get("threads", 0),
        row.get("cpu_mask", "none"),
    )


def compute_median(values: list[float]) -> Optional[float]:
    """Compute median of values, returning None if empty."""
    if not values:
        return None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2


def format_number(val: Optional[float], suffix: str = "") -> str:
    """Format a number with suffix, or '-' if None."""
    if val is None:
        return "-"
    return f"{val:.2f}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare benchmark results from JSONL files."
    )
    parser.add_argument(
        "jsonl_files",
        nargs="+",
        help="One or more JSONL files to compare",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file (default: stdout)",
    )

    args = parser.parse_args()

    all_rows = []
    for path_str in args.jsonl_files:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: File not found: {path}", file=sys.stderr)
            continue
        all_rows.extend(load_jsonl(path))

    if not all_rows:
        print("No data found in input files.", file=sys.stderr)
        sys.exit(1)

    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        key = group_key(row)
        grouped[key].append(row)

    output = sys.stdout
    if args.output:
        output = open(args.output, "w")

    try:
        output.write("# Benchmark Comparison\n\n")

        output.write(
            "| backend | quant | model_id | prompt_id | threads | cpu_mask | "
            "ttft_ms | decode_tok/s | generated | runs | status |\n"
        )
        output.write("|---|---|---|---|---|---|---|---|---|---|---|\n")

        sorted_keys = sorted(grouped.keys())

        for key in sorted_keys:
            backend, quant, model_id, prompt_id, threads, cpu_mask = key
            rows = grouped[key]

            valid_rows = [r for r in rows if r.get("status") == "ok"]
            failed_count = len(rows) - len(valid_rows)

            ttft_values = [
                r["ttft_ms"] for r in valid_rows if r.get("ttft_ms") is not None
            ]
            decode_values = [
                r["decode_tok_per_sec"]
                for r in valid_rows
                if r.get("decode_tok_per_sec") is not None
            ]
            generated_values = [
                r["generated_tokens"]
                for r in valid_rows
                if r.get("generated_tokens") is not None
            ]

            median_ttft = compute_median(ttft_values)
            median_decode = compute_median(decode_values)
            median_generated = (
                compute_median(generated_values) if generated_values else None
            )

            status = "ok" if failed_count == 0 else f"{failed_count} failed"

            output.write(
                f"| {backend} | {quant} | {model_id} | {prompt_id} | "
                f"{threads} | {cpu_mask} | "
                f"{format_number(median_ttft, 'ms')} | "
                f"{format_number(median_decode, 'tok/s')} | "
                f"{format_number(median_generated, '')} | "
                f"{len(valid_rows)} | {status} |\n"
            )

    finally:
        if args.output:
            output.close()


if __name__ == "__main__":
    main()
