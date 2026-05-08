"""
QLoRA Fine-tuning for Llama 3.2 1B Instruct
Loads the base model in 4-bit NF4 quantization (bitsandbytes),
then applies LoRA adapters on top — enabling training on a single 16 GB GPU.
Dataset: cybersecurity_lora_dataset.jsonl
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

from hf_auth import login_huggingface, ensure_model

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = "meta-llama/Llama-3.2-1B-Instruct"
DATA_PATH  = "cybersecurity_lora_dataset.jsonl"
OUTPUT_DIR = "./outputs/qlora"
MAX_LEN    = 512

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",          # NormalFloat4 — optimal for LLMs
    bnb_4bit_compute_dtype    = torch.float16,
    bnb_4bit_use_double_quant = True,           # nested quantization saves ~0.4 bits/param
)

LORA_CONFIG = LoraConfig(
    task_type      = TaskType.CAUSAL_LM,
    r              = 64,
    lora_alpha     = 128,
    lora_dropout   = 0.05,
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    bias           = "none",
)

TRAIN_ARGS = TrainingArguments(
    output_dir              = OUTPUT_DIR,
    num_train_epochs        = 3,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 8,
    learning_rate           = 2e-4,
    lr_scheduler_type       = "cosine",
    warmup_ratio            = 0.05,
    fp16                    = True,
    logging_steps           = 10,
    save_strategy           = "epoch",
    report_to               = "none",
    optim                   = "paged_adamw_8bit",   # 8-bit paged optimizer for QLoRA
    group_by_length         = True,
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
        quantization_config = BNB_CONFIG,
        device_map          = "auto",
        trust_remote_code   = True,
    )
    model = prepare_model_for_kbit_training(model)   # freeze base params, cast norms to fp32
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        args          = TRAIN_ARGS,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length     = MAX_LEN,
        packing            = True,    # pack short sequences to fill context window
    )
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"QLoRA adapter saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
