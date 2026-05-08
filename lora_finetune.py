"""
LoRA Fine-tuning for Llama 3.2 1B Instruct
Dataset: cybersecurity_lora_dataset.jsonl  ({"user": ..., "assistant": ...})
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

from hf_auth import login_huggingface, ensure_model

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = "meta-llama/Llama-3.2-1B-Instruct"
DATA_PATH  = "cybersecurity_lora_dataset.jsonl"
OUTPUT_DIR = "./outputs/lora"
MAX_LEN    = 512

LORA_CONFIG = LoraConfig(
    task_type      = TaskType.CAUSAL_LM,
    r              = 16,          # rank
    lora_alpha     = 32,          # scaling factor
    lora_dropout   = 0.05,
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    bias           = "none",
)

TRAIN_ARGS = TrainingArguments(
    output_dir              = OUTPUT_DIR,
    num_train_epochs        = 3,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    learning_rate           = 2e-4,
    lr_scheduler_type       = "cosine",
    warmup_ratio            = 0.05,
    fp16                    = torch.cuda.is_available(),
    logging_steps           = 10,
    save_strategy           = "epoch",
    report_to               = "none",
    optim                   = "adamw_torch",
)

# ── Data ──────────────────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def format_chat(sample, tokenizer):
    """Format with Llama-3.2 chat template, masking the prompt tokens."""
    messages = [
        {"role": "user",      "content": sample["user"]},
        {"role": "assistant", "content": sample["assistant"]},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    # Prompt-only text (to compute label mask boundary)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": sample["user"]}],
        tokenize=False, add_generation_prompt=True
    )
    enc_full   = tokenizer(full_text,   truncation=True, max_length=MAX_LEN)
    enc_prompt = tokenizer(prompt_text, truncation=True, max_length=MAX_LEN)
    prompt_len = len(enc_prompt["input_ids"])

    labels = enc_full["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len  # mask prompt
    enc_full["labels"] = labels
    return enc_full

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
# ── Auth / model download ─────────────────────────────────────────────────────
login_huggingface()
ensure_model(MODEL_ID)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    raw = load_jsonl(DATA_PATH)
    dataset = Dataset.from_list(raw).map(
        lambda x: format_chat(x, tokenizer),
        remove_columns=["user", "assistant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    trainer = Trainer(
        model         = model,
        args          = TRAIN_ARGS,
        train_dataset = dataset,
        data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
