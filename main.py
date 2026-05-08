#!/usr/bin/env python3
"""
main.py — Fine-tuning launcher for Llama 3.2 1B Instruct
Cybersecurity dataset: cybersecurity_lora_dataset.jsonl

Usage:
    python main.py                      # interactive menu
    python main.py --method qlora       # direct run
    python main.py --method lora --multi-gpu --num-processes 2
"""

import argparse
import os
import subprocess
import sys


def load_env():
    """Load .env from the script directory into os.environ."""
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_file):
        print("[WARN] .env file not found. Create one with HF_TOKEN=hf_your_token_here")
        return
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"\'\"\'\' ')
            if key and value:
                os.environ.setdefault(key, value)

# ── Method registry ───────────────────────────────────────────────────────────
METHODS = {
    "1": ("lora",          "lora_finetune.py",          "LoRA               — Low-Rank Adaptation (~1-2% params, ~8 GB VRAM)"),
    "2": ("qlora",         "qlora_finetune.py",         "QLoRA              — 4-bit quantized LoRA (~1-2% params, ~5-6 GB VRAM) ⭐ recommended"),
    "3": ("sft",           "sft_finetune.py",           "SFT                — Full Supervised Fine-Tuning (100% params, ~20+ GB VRAM)"),
    "4": ("prefix_tuning", "prefix_tuning_finetune.py", "Prefix Tuning      — Virtual token prefix (~0.1% params, ~6 GB VRAM)"),
    "5": ("adapter",       "adapter_tuning_finetune.py","Adapter (IA³)      — Scaling vector adapters (~0.01% params, ~5 GB VRAM)"),
}

NAME_TO_KEY = {v[0]: k for k, v in METHODS.items()}

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     Llama 3.2 1B Instruct — Fine-Tuning Launcher            ║
║     Dataset: cybersecurity_lora_dataset.jsonl               ║
╚══════════════════════════════════════════════════════════════╝
"""

# ── Helpers ───────────────────────────────────────────────────────────────────
def check_dataset():
    path = "cybersecurity_lora_dataset.jsonl"
    if not os.path.exists(path):
        print(f"[ERROR] Dataset not found: {path}")
        print("        Place cybersecurity_lora_dataset.jsonl in the same directory as main.py.")
        sys.exit(1)

def check_script(script_name):
    if not os.path.exists(script_name):
        print(f"[ERROR] Script not found: {script_name}")
        print("        Make sure all finetune scripts are in the same directory as main.py.")
        sys.exit(1)

def print_menu():
    print(BANNER)
    print("  Select a fine-tuning method:\n")
    for key, (_, _, description) in METHODS.items():
        print(f"    [{key}]  {description}")
    print()
    print("    [q]  Quit")
    print()

def build_command(script: str, multi_gpu: bool, num_processes: int, gpu_ids: str) -> list:
    env_prefix = {}
    if gpu_ids:
        env_prefix["CUDA_VISIBLE_DEVICES"] = gpu_ids

    if multi_gpu:
        cmd = [
            "accelerate", "launch",
            "--num_processes", str(num_processes),
            script,
        ]
    else:
        cmd = [sys.executable, script]

    return cmd, env_prefix

def run(script: str, multi_gpu: bool, num_processes: int, gpu_ids: str, method_label: str):
    check_dataset()
    check_script(script)

    cmd, extra_env = build_command(script, multi_gpu, num_processes, gpu_ids)
    env = {**os.environ, **extra_env}

    gpu_note = f"GPU(s): {gpu_ids}" if gpu_ids else "GPU(s): all visible"
    mode_note = f"multi-GPU × {num_processes}" if multi_gpu else "single-GPU"

    token = os.environ.get("HF_TOKEN", "")
    token_note = "✓ found" if token and token != "hf_your_token_here" else "✗ missing — edit .env"
    print(f"\n  Method  : {method_label}")
    print(f"  HF_TOKEN: {token_note}")
    print(f"  Mode    : {mode_note}  |  {gpu_note}")
    print(f"  Command : {' '.join(cmd)}\n")
    print("─" * 64)

    try:
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"\n[ERROR] Process exited with code {result.returncode}")
            sys.exit(result.returncode)
    except FileNotFoundError as e:
        binary = cmd[0]
        if binary == "accelerate":
            print("[ERROR] accelerate not found. Install it: pip install accelerate")
        else:
            print(f"[ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
        sys.exit(0)

# ── Interactive mode ──────────────────────────────────────────────────────────
def interactive():
    print_menu()

    # Select method
    while True:
        choice = input("  Enter choice: ").strip().lower()
        if choice == "q":
            print("Exiting.")
            sys.exit(0)
        if choice in METHODS:
            break
        # allow entering method name directly (e.g. "qlora")
        if choice in NAME_TO_KEY:
            choice = NAME_TO_KEY[choice]
            break
        print("  Invalid choice. Try again.")

    name, script, description = METHODS[choice]

    # Multi-GPU?
    multi_gpu = False
    num_processes = 1
    mg = input("  Use multi-GPU with accelerate? [y/N]: ").strip().lower()
    if mg == "y":
        multi_gpu = True
        try:
            num_processes = int(input("  Number of GPUs/processes [2]: ").strip() or "2")
        except ValueError:
            num_processes = 2

    # Specific GPU IDs?
    gpu_ids = ""
    if not multi_gpu:
        gpu_raw = input("  CUDA_VISIBLE_DEVICES (e.g. 0 or 0,1) [leave blank for default]: ").strip()
        if gpu_raw:
            gpu_ids = gpu_raw

    run(script, multi_gpu, num_processes, gpu_ids, description)

# ── CLI mode ──────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tuning launcher for Llama 3.2 1B Instruct",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # interactive menu
  python main.py --method qlora              # single-GPU QLoRA
  python main.py --method lora --gpus 0,1   # specific GPUs
  python main.py --method sft --multi-gpu --num-processes 4
        """,
    )
    parser.add_argument(
        "--method",
        choices=[v[0] for v in METHODS.values()],
        help="Fine-tuning method to run",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Launch with accelerate for multi-GPU training",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=2,
        help="Number of GPU processes (only with --multi-gpu, default: 2)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help="CUDA_VISIBLE_DEVICES value (e.g. '0' or '0,1')",
    )
    return parser.parse_args()

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    load_env()
    args = parse_args()

    if args.method is None:
        # No --method flag: show interactive menu
        interactive()
    else:
        key = NAME_TO_KEY[args.method]
        name, script, description = METHODS[key]
        run(script, args.multi_gpu, args.num_processes, args.gpus, description)

if __name__ == "__main__":
    main()
