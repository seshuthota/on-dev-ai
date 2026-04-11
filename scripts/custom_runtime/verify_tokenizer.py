#!/usr/bin/env python3
"""
Verify tokenizer output against HuggingFace transformers AutoTokenizer.
This is the oracle validation for the custom runtime tokenizer.

Usage:
    python3 verify_tokenizer.py [--run-reference PATH] [--prompts FILE]

The run_reference tool should output JSON array of token IDs like:
    [1, 15043, 3186]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_MODEL_DIR = "/home/curious/models/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_RUN_REFERENCE = "./native/build-llama/run_tokenizer"

TEST_PROMPTS = [
    "Hello",
    "Hello world",
    "The quick brown fox",
    "  hello  ",
    "Hello!world?",
    "a b c d e f g",
]


def get_hf_tokens(model_dir: str, text: str, add_bos: bool = False) -> list[int]:
    """Get tokens from HuggingFace AutoTokenizer."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if add_bos:
        tokens = [tokenizer.bos_token_id] + tokens

    return tokens


def get_cpp_tokens(run_tokenizer_path: str, vocab_bin_path: str, text: str, add_bos: bool = False) -> list[int]:
    """Get tokens from the C++ run_tokenizer tool."""
    cmd = [run_tokenizer_path, vocab_bin_path]
    if add_bos:
        cmd.append("--add-bos")
    cmd.append(text)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: run_tokenizer failed: {result.stderr}")
        return []

    # Parse JSON array output like "[1, 15043, 3186]"
    output = result.stdout.strip()
    if output.startswith("["):
        # Extract numbers from "[1, 15043, 3186]"
        nums = output[1:-1].split(",")
        return [int(n.strip()) for n in nums if n.strip()]

    print(f"ERROR: Could not parse output: {output}")
    return []


def main():
    parser = argparse.ArgumentParser(description="Verify custom runtime tokenizer against HF oracle")
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="HuggingFace model directory"
    )
    parser.add_argument(
        "--vocab-bin",
        default=".artifacts/custom-runtime/tinyllama-v1/tokenizer/vocab.bin",
        help="Path to vocab.bin"
    )
    parser.add_argument(
        "--run-tokenizer",
        default=DEFAULT_RUN_REFERENCE,
        help="Path to run_tokenizer executable"
    )
    parser.add_argument(
        "--add-bos",
        action="store_true",
        help="Include BOS token"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )
    args = parser.parse_args()

    vocab_bin_path = Path(args.vocab_bin)
    if not vocab_bin_path.is_absolute():
        vocab_bin_path = Path.cwd() / vocab_bin_path

    run_tokenizer_path = Path(args.run_tokenizer)
    if not run_tokenizer_path.is_absolute():
        run_tokenizer_path = Path.cwd() / run_tokenizer_path

    if not run_tokenizer_path.exists():
        print(f"ERROR: run_tokenizer not found at {run_tokenizer_path}")
        print("Hint: Run 'cd native/build-llama && make run_tokenizer' to build it")
        sys.exit(1)

    if not vocab_bin_path.exists():
        print(f"ERROR: vocab.bin not found at {vocab_bin_path}")
        print("Hint: Run 'python3 scripts/custom_runtime/pack_tinyllama.py' to create it")
        sys.exit(1)

    print(f"Model: {args.model_dir}")
    print(f"Vocab bin: {vocab_bin_path}")
    print(f"Run tokenizer: {run_tokenizer_path}")
    print(f"Add BOS: {args.add_bos}")
    print()

    passed = 0
    failed = 0

    for prompt in TEST_PROMPTS:
        hf_tokens = get_hf_tokens(args.model_dir, prompt, add_bos=args.add_bos)
        cpp_tokens = get_cpp_tokens(str(run_tokenizer_path), str(vocab_bin_path), prompt, add_bos=args.add_bos)

        if hf_tokens == cpp_tokens:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1

        print(f"[{status}] prompt={repr(prompt)}")
        if args.verbose or status == "FAIL":
            print(f"  HF tokens:  {hf_tokens}")
            print(f"  C++ tokens: {cpp_tokens}")
            if hf_tokens != cpp_tokens:
                print(f"  HF length: {len(hf_tokens)}, C++ length: {len(cpp_tokens)}")

    print()
    print(f"Results: {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()