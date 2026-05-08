"""
SFT (Full Fine-tuning) — Llama 3.2 1B Instruct
Dataset : cybersecurity_lora_dataset.jsonl
Method  : All parameters updated. Highest quality, highest VRAM.
VRAM    : ~20+ GB
"""

import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from hf_auth import login_huggingface, ensure_model

MODEL_ID   = "meta-llama/Llama-3.2-1B-Instruct"
DATA_PATH  = "cybersecurity_lora_dataset.jsonl"
OUTPUT_DIR = "./outputs/sft"

_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
_fp16 = torch.cuda.is_available() and not _bf16

SFT_CFG = SFTConfig(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = 3,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    gradient_checkpointing      = True,
    learning_rate               = 2e-5,
    lr_scheduler_type           = "cosine",
    warmup_steps                = 10,
    bf16                        = _bf16,
    fp16                        = _fp16,
    logging_steps               = 10,
    save_strategy               = "epoch",
    report_to                   = "none",
    optim                       = "adamw_torch_fused",
    weight_decay                = 0.01,
    # SFT-specific
    max_length                  = 512,
    dataset_text_field          = "text",
    packing                     = False,
)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def format_text(sample, tokenizer):
    messages = [{"role": "user",      "content": sample["user"]},
                {"role": "assistant", "content": sample["assistant"]}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main():
    login_huggingface()
    ensure_model(MODEL_ID)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list(load_jsonl(DATA_PATH)).map(
        lambda x: {"text": format_text(x, tokenizer)},
        remove_columns=["user", "assistant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype = torch.bfloat16 if _bf16 else torch.float32,
        device_map  = "auto",
    )

    SFTTrainer(
        model            = model,
        processing_class = tokenizer,
        args             = SFT_CFG,
        train_dataset    = dataset,
    ).train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"SFT model saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
