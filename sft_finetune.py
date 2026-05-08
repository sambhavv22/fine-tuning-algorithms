"""
Supervised Fine-Tuning (SFT) for Llama 3.2 1B Instruct
Full-parameter fine-tuning via HuggingFace TRL SFTTrainer.
For memory-constrained setups, enable gradient checkpointing and use bf16.
Dataset: cybersecurity_lora_dataset.jsonl
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import SFTTrainer

from hf_auth import login_huggingface, ensure_model

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = "meta-llama/Llama-3.2-1B-Instruct"
DATA_PATH  = "cybersecurity_lora_dataset.jsonl"
OUTPUT_DIR = "./outputs/sft"
MAX_LEN    = 512

TRAIN_ARGS = TrainingArguments(
    output_dir              = OUTPUT_DIR,
    num_train_epochs        = 3,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    gradient_checkpointing  = True,   # trade compute for memory
    learning_rate           = 2e-5,   # lower LR for full fine-tune
    lr_scheduler_type       = "cosine",
    warmup_ratio            = 0.05,
    bf16                    = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
    fp16                    = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
    logging_steps           = 10,
    save_strategy           = "epoch",
    report_to               = "none",
    optim                   = "adamw_torch_fused",
    weight_decay            = 0.01,
)

# ── Data ──────────────────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def format_prompt(sample, tokenizer):
    messages = [
        {"role": "user",      "content": sample["user"]},
        {"role": "assistant", "content": sample["assistant"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
# ── Auth / model download ─────────────────────────────────────────────────────
login_huggingface()
ensure_model(MODEL_ID)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    raw     = load_jsonl(DATA_PATH)
    dataset = Dataset.from_list(raw)
    dataset = dataset.map(
        lambda x: {"text": format_prompt(x, tokenizer)},
        remove_columns=["user", "assistant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map  = "auto",
        trust_remote_code = True,
    )

    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        args               = TRAIN_ARGS,
        train_dataset      = dataset,
        dataset_text_field = "text",
        max_seq_length     = MAX_LEN,
        packing            = False,
    )
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"SFT model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
